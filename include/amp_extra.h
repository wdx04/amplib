#pragma once

// TODO List
// gftt, harris, fast, orb, sift, surf, bfmatcher, svm, match template,
// distance transform, watershed, grabcut, kmeans, knn, hough circles, generalized hough

#include "amp_core.h"
#include "amp_gaussian.h"

namespace amp
{
	// Area of maximum zero rectangle
	inline int max_allzero_area_32f_c1(concurrency::accelerator_view& acc_view, concurrency::array_view<const float, 2> src_array, concurrency::array_view<float, 2> temp_array)
	{
		int rows = src_array.get_extent()[0];
		int cols = src_array.get_extent()[1];
		// 1st pass
		parallel_for_each(acc_view, concurrency::extent<1>(cols), [=](concurrency::index<1> idx) restrict(amp)
		{
			int col = idx[0];
			float count = 0.0f;
			for(int i = 0; i < rows; i++)
			{
				count = src_array(i, col) == 0.0f ? count + 1.0f : 0.0f;
				temp_array(i, col) = count;
			}
		});
		// 2nd pass
		concurrency::array<int, 1> max_areas(rows, acc_view);
		parallel_for_each(acc_view, concurrency::extent<1>(rows), [=, &max_areas](concurrency::index<1> idx) restrict(amp)
		{
			int row = idx[0];
			array_view<const float, 1> row_data = temp_array[row];
			int_2 stack[256];
			int stack_ptr = -1;
			int max_area = 0;
			int pos = 0;
			for(; pos < cols; pos++)
			{
				int start = pos;
				int height = int(row_data[pos]);
				while(true)
				{
					if(stack_ptr < 0 || height > stack[stack_ptr].y)
					{
						stack[++stack_ptr] = int_2(start, height);
					}
					else if(stack_ptr >= 0 && height < stack[stack_ptr].y)
					{
						max_area = direct3d::imax(max_area, stack[stack_ptr].y * (pos - stack[stack_ptr].x));
						start = stack[stack_ptr--].x;
						continue;
					}
					break;
				}
				if(stack_ptr == 255)
				{
					// stack would overflow
					break;
				}
			}
			for(int i = 0; i <= stack_ptr; i++)
			{
				max_area = direct3d::imax(max_area, stack[i].y * (pos - stack[i].x));
			}
			max_areas[row] = max_area;
		});
		std::vector<int> cpu_max_areas(rows);
		concurrency::copy(max_areas, cpu_max_areas.begin());
		return *std::max_element(cpu_max_areas.begin(), cpu_max_areas.end());
	}

	inline void illuminance_correction_32f_c1(concurrency::accelerator_view& acc_view, concurrency::array_view<const float, 2> src_array
		, concurrency::array_view<float, 2> dest_array, concurrency::array_view<float, 2> temp_array1, float opacity = 0.5f)
	{
		int kernel_size = (std::min<int>(src_array.get_extent()[0], src_array.get_extent()[1]) / 2 - 1) | 1;
		float sigma = float(kernel_size) / 6.0f;
		// 1.gaussian blur
		gaussian_filter_32f_c1(acc_view, src_array, dest_array, temp_array1, kernel_size, sigma);
		// 2.correct
		static const int tile_size = 32;
		const float center_val = 128.0f;
		concurrency::parallel_for_each(acc_view, temp_array1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float original_val = guarded_read(src_array, idx.global);
			float gauss_val = guarded_read(dest_array, idx.global);
			guarded_write(dest_array, idx.global, (1.0f - (gauss_val - center_val) / center_val * opacity) * original_val);
		});
	}

	inline void illuminance_correction_32f_c3(concurrency::accelerator_view& acc_view
		, concurrency::array_view<const float, 2> src_array_ch0, concurrency::array_view<const float, 2> src_array_ch1
		, concurrency::array_view<const float, 2> src_array_ch2, concurrency::array_view<const float, 2> src_array_ch3
		, concurrency::array_view<float, 2> dest_array_ch1, concurrency::array_view<float, 2> dest_array_ch2
		, concurrency::array_view<float, 2> dest_array_ch3, concurrency::array_view<float, 2> temp_array1, float opacity = 0.5f)
	{
		int kernel_size = (std::min<int>(src_array_ch0.get_extent()[0], src_array_ch0.get_extent()[1]) / 2 - 1) | 1;
		float sigma = float(kernel_size) / 6.0f;
		// 1.gaussian blur
		gaussian_filter_32f_c1(acc_view, src_array_ch0, dest_array_ch1, dest_array_ch2, kernel_size, sigma);
		// 2.correct
		static const int tile_size = 32;
		const float center_val = 128.0f;
		concurrency::parallel_for_each(acc_view, temp_array1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float original_val1 = guarded_read(src_array_ch1, idx.global);
			float original_val2 = guarded_read(src_array_ch2, idx.global);
			float original_val3 = guarded_read(src_array_ch3, idx.global);
			float gauss_val = guarded_read(dest_array_ch1, idx.global);
			guarded_write(dest_array_ch1, idx.global, (1.0f - (gauss_val - center_val) / center_val * opacity) * original_val1);
			guarded_write(dest_array_ch2, idx.global, (1.0f - (gauss_val - center_val) / center_val * opacity) * original_val2);
			guarded_write(dest_array_ch3, idx.global, (1.0f - (gauss_val - center_val) / center_val * opacity) * original_val3);
		});
	}
}
