#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include "amp_core.h"
#include "amp_calc_hist.h"

namespace amp
{
	inline float get_otsu_thresh(accelerator_view& acc_view, array_view<const float, 2> src_array)
	{
		static const int N = 256;
		// 1.calculate histogram
		concurrency::array<int, 1> hist_array(N, acc_view);
		calc_hist_32f_c1(acc_view, src_array, hist_array, N);
		std::vector<int> h(N);
		concurrency::copy(hist_array, h.begin());
		// 2.otsu
		float mu = 0.0, scale = 1.0f / (src_array.get_extent()[0] * src_array.get_extent()[1]);
		for(int i = 0; i < N; i++)
			mu += i * (float)h[i];
		mu *= scale;
		float mu1 = 0.0f, q1 = 0.0f;
		float max_sigma = 0.0f, max_val = 0.0f;
		for(int i = 0; i < N; i++)
		{
			float p_i, q2, mu2, sigma;
			p_i = h[i] * scale;
			mu1 *= q1;
			q1 += p_i;
			q2 = 1.0f - q1;
			if(std::min<float>(q1, q2) < FLT_EPSILON || std::max<float>(q1, q2) > 1. - FLT_EPSILON)
				continue;
			mu1 = (mu1 + i*p_i) / q1;
			mu2 = (mu - q1*mu1) / q2;
			sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
			if(sigma > max_sigma)
			{
				max_sigma = sigma;
				max_val = (float)i;
			}
		}
		return max_val;
	}

	inline float threshold_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, float thresh, float max_value, int type)
	{
		if(type & cv::THRESH_OTSU)
		{
			thresh = get_otsu_thresh(acc_view, src_array);
			type &= ~cv::THRESH_OTSU;
		}
		dest_array.discard_data();
		static const int tile_size = 32;
		switch(type)
		{
		case cv::THRESH_BINARY_INV:
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val = guarded_read(src_array, idx.global);
				guarded_write(dest_array, idx.global, src_val > thresh ? 0.0f : max_value);
			});
			break;
		case cv::THRESH_TRUNC:
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val = guarded_read(src_array, idx.global);
				guarded_write(dest_array, idx.global, src_val > thresh ? thresh : src_val);
			});
			break;
		case cv::THRESH_TOZERO:
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val = guarded_read(src_array, idx.global);
				guarded_write(dest_array, idx.global, src_val > thresh ? src_val : 0.0f);
			});
			break;
		case cv::THRESH_TOZERO_INV:
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val = guarded_read(src_array, idx.global);
				guarded_write(dest_array, idx.global, src_val > thresh ? 0.0f : src_val);
			});
			break;
		case cv::THRESH_BINARY:
		default:
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val = guarded_read(src_array, idx.global);
				guarded_write(dest_array, idx.global, src_val > thresh ? max_value : 0.0f);
			});
			break;
		}
		return thresh;
	}
}
