#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include "amp_core.h"
#include "amp_conv_separable.h"

namespace amp
{
	// Gaussian Blur Filter
	inline void gaussian_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> row_temp_array
		, int ksize, float sigma)
	{
		if(ksize <= 0 && sigma > 0.0f)
		{
			ksize = cvRound(sigma * 6 + 1) | 1;
		}
		cv::Mat k = cv::getGaussianKernel(ksize, sigma, CV_32F);
		convolve_separable_32f_c1(acc_view, src_array, dest_array, row_temp_array, k, k);
	}

	// Recursive Gaussian Blur Filter
	inline float iir_kernel(float in, float out_1, float out_2, float out_3, const float* coefs) restrict(amp)
	{
		return coefs[0] * in + coefs[1] * out_1 + coefs[2] * out_2 + coefs[3] * out_3;
	}

	// max_pad >= max(sigma * 3, 3);
	template<int max_pad>
	inline void recursive_gaussian_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, float sigma, int pad)
	{
		float q = sigma < 0.5f ? 0.1147705f : (sigma < 2.5f ? 3.97156f - 4.14554f * std::sqrtf(1.0f - 0.26891f * sigma) : 0.98711f * sigma - 0.96330f);
		float b0 = 1.57825f + q * (2.44413f + q * (1.42810f + q * 0.422205f));
		float b1 = q * (2.44413f + q * (2.85619f + q * 1.26661f));
		float b2 = q * q * (-1.42810f + q * -1.26661f);
		float b3 = q * q * q * 0.422205f;
		kernel_wrapper<float, 4U> coefs(1, 4, { 1.0f - (b1 + b2 + b3) / b0, b1 / b0, b2 / b0, b3 / b0 });
		pad = std::min(max_pad, pad);
		const static int tile_size = 32;
		// row pass
		parallel_for_each(acc_view, concurrency::extent<1>(src_array.get_extent()[0]).tile<tile_size>().pad(), [=](concurrency::tiled_index<tile_size> idx) restrict(amp)
		{
			if(idx.global[0] < src_array.get_extent()[0])
			{
				int cols = src_array.get_extent()[1];
				int i;
				array_view<const float, 1> row = src_array[idx.global[0]];
				array_view<float, 1> dest_row = dest_array[idx.global[0]];
				float z[max_pad];
				// forward transform
				z[0] = iir_kernel(row[pad - 1], 0.0f, 0.0f, 0.0f, coefs.data);
				z[1] = iir_kernel(row[pad - 2], z[0], 0.0f, 0.0f, coefs.data);
				z[2] = iir_kernel(row[pad - 3], z[1], z[0], 0.0f, coefs.data);
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(pad, 4); i += 4)
				{
					z[i] = iir_kernel(row[pad - i - 1], z[i - 1], z[i - 2], z[i - 3], coefs.data);
					z[i + 1] = iir_kernel(row[pad - i - 2], z[i], z[i - 1], z[i - 2], coefs.data);
					z[i + 2] = iir_kernel(row[pad - i - 3], z[i + 1], z[i], z[i - 1], coefs.data);
					z[i + 3] = iir_kernel(row[pad - i - 4], z[i + 2], z[i + 1], z[i], coefs.data);
				}
#endif
				for(; i < pad; ++i)
				{
					z[i] = iir_kernel(row[pad - i - 1], z[i - 1], z[i - 2], z[i - 3], coefs.data);
				}
				dest_row[0] = iir_kernel(row[0], z[pad - 1], z[pad - 2], z[pad - 3], coefs.data);
				dest_row[1] = iir_kernel(row[1], dest_row[0], z[pad - 1], z[pad - 2], coefs.data);
				dest_row[2] = iir_kernel(row[2], dest_row[1], dest_row[0], z[pad - 1], coefs.data);
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i + pad < ROUNDDOWN(cols, 4); i += 4)
				{
					dest_row[i] = iir_kernel(row[i], dest_row[i - 1], dest_row[i - 2], dest_row[i - 3], coefs.data);
					dest_row[i + 1] = iir_kernel(row[i + 1], dest_row[i], dest_row[i - 1], dest_row[i - 2], coefs.data);
					dest_row[i + 2] = iir_kernel(row[i + 2], dest_row[i + 1], dest_row[i], dest_row[i - 1], coefs.data);
					dest_row[i + 3] = iir_kernel(row[i + 3], dest_row[i + 2], dest_row[i + 1], dest_row[i], coefs.data);
				}
#endif
				for(; i + pad < cols; ++i)
				{
					dest_row[i] = iir_kernel(row[i], dest_row[i - 1], dest_row[i - 2], dest_row[i - 3], coefs.data);
				}
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(cols, 4); i += 4)
				{
					z[i + pad - cols] = row[i];
					dest_row[i] = iir_kernel(row[i], dest_row[i - 1], dest_row[i - 2], dest_row[i - 3], coefs.data);
					z[i + 1 + pad - cols] = row[i + 1];
					dest_row[i + 1] = iir_kernel(row[i + 1], dest_row[i], dest_row[i - 1], dest_row[i - 2], coefs.data);
					z[i + 2 + pad - cols] = row[i + 2];
					dest_row[i + 2] = iir_kernel(row[i + 2], dest_row[i + 1], dest_row[i], dest_row[i - 1], coefs.data);
					z[i + 3 + pad - cols] = row[i + 3];
					dest_row[i + 3] = iir_kernel(row[i + 3], dest_row[i + 2], dest_row[i + 1], dest_row[i], coefs.data);
				}
#endif
				for(; i < cols; ++i)
				{
					z[i + pad - cols] = row[i];
					dest_row[i] = iir_kernel(row[i], dest_row[i - 1], dest_row[i - 2], dest_row[i - 3], coefs.data);
				}
				// backward transform
				z[pad - 1] = iir_kernel(z[pad - 1], dest_row[cols - 1], dest_row[cols - 2], dest_row[cols - 3], coefs.data);
				z[pad - 2] = iir_kernel(z[pad - 2], z[pad - 1], dest_row[cols - 1], dest_row[cols - 2], coefs.data);
				z[pad - 3] = iir_kernel(z[pad - 3], z[pad - 2], z[pad - 1], dest_row[cols - 1], coefs.data);
				i = pad - 4;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i >= ROUNDDOWN(pad - 4, 4); --i)
				{
					z[i] = iir_kernel(z[i], z[i + 1], z[i + 2], z[i + 3], coefs.data);
				}
#endif
				for(; i >= 0; i -= 4)
				{
					z[i] = iir_kernel(z[i], z[i + 1], z[i + 2], z[i + 3], coefs.data);
					z[i - 1] = iir_kernel(z[i - 1], z[i], z[i + 1], z[i + 2], coefs.data);
					z[i - 2] = iir_kernel(z[i - 2], z[i - 1], z[i], z[i + 1], coefs.data);
					z[i - 3] = iir_kernel(z[i - 3], z[i - 2], z[i - 1], z[i], coefs.data);
				}
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(pad, 4); i += 4)
				{
					z[i] = iir_kernel(z[i], z[i - 1], z[i - 2], z[i - 3], coefs.data);
					z[i + 1] = iir_kernel(z[i + 1], z[i], z[i - 1], z[i - 2], coefs.data);
					z[i + 2] = iir_kernel(z[i + 2], z[i + 1], z[i], z[i - 1], coefs.data);
					z[i + 3] = iir_kernel(z[i + 3], z[i + 2], z[i + 1], z[i], coefs.data);
				}
#endif
				for(; i < pad; ++i)
				{
					z[i] = iir_kernel(z[i], z[i - 1], z[i - 2], z[i - 3], coefs.data);
				}
				dest_row[cols - 1] = iir_kernel(dest_row[cols - 1], z[pad - 1], z[pad - 2], z[pad - 3], coefs.data);
				dest_row[cols - 2] = iir_kernel(dest_row[cols - 2], dest_row[cols - 1], z[pad - 1], z[pad - 2], coefs.data);
				dest_row[cols - 3] = iir_kernel(dest_row[cols - 3], dest_row[cols - 2], dest_row[cols - 1], z[pad - 1], coefs.data);
				i = cols - 4;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i >= ROUNDDOWN(cols - 4, 4); --i)
				{
					dest_row[i] = iir_kernel(dest_row[i], dest_row[i + 1], dest_row[i + 2], dest_row[i + 3], coefs.data);
				}
#endif
				for(; i >= 0; i -= 4)
				{
					dest_row[i] = iir_kernel(dest_row[i], dest_row[i + 1], dest_row[i + 2], dest_row[i + 3], coefs.data);
					dest_row[i - 1] = iir_kernel(dest_row[i - 1], dest_row[i], dest_row[i + 1], dest_row[i + 2], coefs.data);
					dest_row[i - 2] = iir_kernel(dest_row[i - 2], dest_row[i - 1], dest_row[i], dest_row[i + 1], coefs.data);
					dest_row[i - 3] = iir_kernel(dest_row[i - 3], dest_row[i - 2], dest_row[i - 1], dest_row[i], coefs.data);
				}
			}
		});
		// column pass
		parallel_for_each(acc_view, concurrency::extent<1>(dest_array.get_extent()[1]).tile<tile_size>().pad(), [=](concurrency::tiled_index<tile_size> idx) restrict(amp)
		{
			int rows = dest_array.get_extent()[0];
			int col = idx.global[0];
			if(col < dest_array.get_extent()[1])
			{
				int i;
				float z[max_pad];
				// forward transform
				z[0] = iir_kernel(dest_array(pad - 1, col), 0.0f, 0.0f, 0.0f, coefs.data);
				z[1] = iir_kernel(dest_array(pad - 2, col), z[0], 0.0f, 0.0f, coefs.data);
				z[2] = iir_kernel(dest_array(pad - 3, col), z[1], z[0], 0.0f, coefs.data);
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(pad, 4); i += 4)
				{
					z[i] = iir_kernel(dest_array(pad - i - 1, col), z[i - 1], z[i - 2], z[i - 3], coefs.data);
					z[i + 1] = iir_kernel(dest_array(pad - i - 2, col), z[i], z[i - 1], z[i - 2], coefs.data);
					z[i + 2] = iir_kernel(dest_array(pad - i - 3, col), z[i + 1], z[i], z[i - 1], coefs.data);
					z[i + 3] = iir_kernel(dest_array(pad - i - 4, col), z[i + 2], z[i + 1], z[i], coefs.data);
				}
#endif
				for(; i < pad; ++i)
				{
					z[i] = iir_kernel(dest_array(pad - i - 1, col), z[i - 1], z[i - 2], z[i - 3], coefs.data);
				}
				dest_array(0, col) = iir_kernel(dest_array(0, col), z[pad - 1], z[pad - 2], z[pad - 3], coefs.data);
				dest_array(1, col) = iir_kernel(dest_array(1, col), dest_array(0, col), z[pad - 1], z[pad - 2], coefs.data);
				dest_array(2, col) = iir_kernel(dest_array(2, col), dest_array(1, col), dest_array(0, col), z[pad - 1], coefs.data);
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i + pad < ROUNDDOWN(rows, 4); i += 4)
				{
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), dest_array(i - 3, col), coefs.data);
					dest_array(i + 1, col) = iir_kernel(dest_array(i + 1, col), dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), coefs.data);
					dest_array(i + 2, col) = iir_kernel(dest_array(i + 2, col), dest_array(i + 1, col), dest_array(i, col), dest_array(i - 1, col), coefs.data);
					dest_array(i + 3, col) = iir_kernel(dest_array(i + 3, col), dest_array(i + 2, col), dest_array(i + 1, col), dest_array(i, col), coefs.data);
				}
#endif
				for(; i + pad < rows; ++i)
				{
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), dest_array(i - 3, col), coefs.data);
				}
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(rows, 4); i += 4)
				{
					z[i + pad - rows] = dest_array(i, col);
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), dest_array(i - 3, col), coefs.data);
					z[i + 1 + pad - rows] = dest_array(i + 1, col);
					dest_array(i + 1, col) = iir_kernel(dest_array(i + 1, col), dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), coefs.data);
					z[i + 2 + pad - rows] = dest_array(i + 2, col);
					dest_array(i + 2, col) = iir_kernel(dest_array(i + 2, col), dest_array(i + 1, col), dest_array(i, col), dest_array(i - 1, col), coefs.data);
					z[i + 3 + pad - rows] = dest_array(i + 3, col);
					dest_array(i + 3, col) = iir_kernel(dest_array(i + 3, col), dest_array(i + 2, col), dest_array(i + 1, col), dest_array(i, col), coefs.data);
				}
#endif
				for(; i < rows; ++i)
				{
					z[i + pad - rows] = dest_array(i, col);
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i - 1, col), dest_array(i - 2, col), dest_array(i - 3, col), coefs.data);
				}
				// backward transform
				z[pad - 1] = iir_kernel(z[pad - 1], dest_array(rows - 1, col), dest_array(rows - 2, col), dest_array(rows - 3, col), coefs.data);
				z[pad - 2] = iir_kernel(z[pad - 2], z[pad - 1], dest_array(rows - 1, col), dest_array(rows - 2, col), coefs.data);
				z[pad - 3] = iir_kernel(z[pad - 3], z[pad - 2], z[pad - 1], dest_array(rows - 1, col), coefs.data);
				i = pad - 4;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i >= ROUNDDOWN(pad - 4, 4); --i)
				{
					z[i] = iir_kernel(z[i], z[i + 1], z[i + 2], z[i + 3], coefs.data);
				}
#endif
				for(; i >= 0; i -= 4)
				{
					z[i] = iir_kernel(z[i], z[i + 1], z[i + 2], z[i + 3], coefs.data);
					z[i - 1] = iir_kernel(z[i - 1], z[i], z[i + 1], z[i + 2], coefs.data);
					z[i - 2] = iir_kernel(z[i - 2], z[i - 1], z[i], z[i + 1], coefs.data);
					z[i - 3] = iir_kernel(z[i - 3], z[i - 2], z[i - 1], z[i], coefs.data);
				}
				i = 3;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i < ROUNDDOWN(pad, 4); i += 4)
				{
					z[i] = iir_kernel(z[i], z[i - 1], z[i - 2], z[i - 3], coefs.data);
					z[i + 1] = iir_kernel(z[i + 1], z[i], z[i - 1], z[i - 2], coefs.data);
					z[i + 2] = iir_kernel(z[i + 2], z[i + 1], z[i], z[i - 1], coefs.data);
					z[i + 3] = iir_kernel(z[i + 3], z[i + 2], z[i + 1], z[i], coefs.data);
				}
#endif
				for(; i < pad; ++i)
				{
					z[i] = iir_kernel(z[i], z[i - 1], z[i - 2], z[i - 3], coefs.data);
				}
				dest_array(rows - 1, col) = iir_kernel(dest_array(rows - 1, col), z[pad - 1], z[pad - 2], z[pad - 3], coefs.data);
				dest_array(rows - 2, col) = iir_kernel(dest_array(rows - 2, col), dest_array(rows - 1, col), z[pad - 1], z[pad - 2], coefs.data);
				dest_array(rows - 3, col) = iir_kernel(dest_array(rows - 3, col), dest_array(rows - 2, col), dest_array(rows - 1, col), z[pad - 1], coefs.data);
				i = rows - 4;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
				for(; i >= ROUNDDOWN(rows - 4, 4); --i)
				{
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i + 1, col), dest_array(i + 2, col), dest_array(i + 3, col), coefs.data);
				}
#endif
				for(; i >= 0; i -= 4)
				{
					dest_array(i, col) = iir_kernel(dest_array(i, col), dest_array(i + 1, col), dest_array(i + 2, col), dest_array(i + 3, col), coefs.data);
					dest_array(i - 1, col) = iir_kernel(dest_array(i - 1, col), dest_array(i, col), dest_array(i + 1, col), dest_array(i + 2, col), coefs.data);
					dest_array(i - 2, col) = iir_kernel(dest_array(i - 2, col), dest_array(i - 1, col), dest_array(i, col), dest_array(i + 1, col), coefs.data);
					dest_array(i - 3, col) = iir_kernel(dest_array(i - 3, col), dest_array(i - 2, col), dest_array(i - 1, col), dest_array(i, col), coefs.data);
				}
			}
		});
	}

}
