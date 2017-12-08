#pragma once

#include "amp_core.h"

namespace amp
{
	// Hit or miss transform
	template<unsigned int KSize>
	inline void hit_or_miss_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const kernel_wrapper<float, KSize>& kernel)
	{
		static const int tile_size = 32;
		static const int tile_static_size_3x3 = tile_size + 2;
		static const int tile_static_size_5x5 = tile_size + 4;
		static const int max_1d_tile_static_size = tile_size * tile_size * 2;
		assert((tile_size + (kernel.cols & ~1)) * (tile_size + (kernel.rows & ~1)) <= max_1d_tile_static_size);
		if(kernel.rows == 3 && kernel.cols == 3)
		{
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size_3x3][tile_static_size_3x3];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - 1;
				int dy = col - local_col - 1;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_3x3 * tile_static_size_3x3 / 2 - 1);
				int dr = id / tile_static_size_3x3;
				int dc = id % tile_static_size_3x3;
				data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), FLT_MAX);
				data[dr + tile_static_size_3x3 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();

				float result_fg = FLT_MAX;
				float result_bg = FLT_MAX;
				if(kernel.data[0] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 0]); }
				else if(kernel.data[0] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 0]); }
				if(kernel.data[1] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 1]); }
				else if(kernel.data[1] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 1]); }
				if(kernel.data[2] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 2]); }
				else if(kernel.data[2] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 2]); }
				if(kernel.data[3] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 0]); }
				else if(kernel.data[3] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 0]); }
				if(kernel.data[4] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 1]); }
				else if(kernel.data[4] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 1]); }
				if(kernel.data[5] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 2]); }
				else if(kernel.data[5] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 2]); }
				if(kernel.data[6] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 0]); }
				else if(kernel.data[6] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 0]); }
				if(kernel.data[7] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 1]); }
				else if(kernel.data[7] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 1]); }
				if(kernel.data[8] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 2]); }
				else if(kernel.data[8] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 2]); }
				guarded_write(dest_array, idx.global, fast_math::fminf(result_fg, result_bg));
			});
		}
		else if(kernel.rows == 5 && kernel.cols == 5)
		{
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size_5x5][tile_static_size_5x5];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - 2;
				int dy = col - local_col - 2;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_5x5 * tile_static_size_5x5 / 2 - 1);
				int dr = id / tile_static_size_5x5;
				int dc = id % tile_static_size_5x5;
				data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), FLT_MAX);
				data[dr + tile_static_size_5x5 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_5x5 / 2, dy + dc), FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();

				float result_fg = FLT_MAX;
				float result_bg = FLT_MAX;
				if(kernel.data[0] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 0]); }
				else if(kernel.data[0] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 0]); }
				if(kernel.data[1] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 1]); }
				else if(kernel.data[1] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 1]); }
				if(kernel.data[2] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 2]); }
				else if(kernel.data[2] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 2]); }
				if(kernel.data[3] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 3]); }
				else if(kernel.data[3] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 3]); }
				if(kernel.data[4] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 0][local_col + 4]); }
				else if(kernel.data[4] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 0][local_col + 4]); }
				if(kernel.data[5] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 0]); }
				else if(kernel.data[5] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 0]); }
				if(kernel.data[6] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 1]); }
				else if(kernel.data[6] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 1]); }
				if(kernel.data[7] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 2]); }
				else if(kernel.data[7] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 2]); }
				if(kernel.data[8] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 3]); }
				else if(kernel.data[8] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 3]); }
				if(kernel.data[9] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 1][local_col + 4]); }
				else if(kernel.data[9] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 1][local_col + 4]); }
				if(kernel.data[10] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 0]); }
				else if(kernel.data[10] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 0]); }
				if(kernel.data[11] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 1]); }
				else if(kernel.data[11] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 1]); }
				if(kernel.data[12] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 2]); }
				else if(kernel.data[12] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 2]); }
				if(kernel.data[13] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 3]); }
				else if(kernel.data[13] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 3]); }
				if(kernel.data[14] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 2][local_col + 4]); }
				else if(kernel.data[14] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 2][local_col + 4]); }
				if(kernel.data[15] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 3][local_col + 0]); }
				else if(kernel.data[15] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 3][local_col + 0]); }
				if(kernel.data[16] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 3][local_col + 1]); }
				else if(kernel.data[16] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 3][local_col + 1]); }
				if(kernel.data[17] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 3][local_col + 2]); }
				else if(kernel.data[17] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 3][local_col + 2]); }
				if(kernel.data[18] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 3][local_col + 3]); }
				else if(kernel.data[18] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 3][local_col + 3]); }
				if(kernel.data[19] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 3][local_col + 4]); }
				else if(kernel.data[19] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 3][local_col + 4]); }
				if(kernel.data[20] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 4][local_col + 0]); }
				else if(kernel.data[20] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 4][local_col + 0]); }
				if(kernel.data[21] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 4][local_col + 1]); }
				else if(kernel.data[21] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 4][local_col + 1]); }
				if(kernel.data[22] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 4][local_col + 2]); }
				else if(kernel.data[22] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 4][local_col + 2]); }
				if(kernel.data[23] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 4][local_col + 3]); }
				else if(kernel.data[23] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 4][local_col + 3]); }
				if(kernel.data[24] > 0.0f) { result_fg = fast_math::fminf(result_fg, data[local_row + 4][local_col + 4]); }
				else if(kernel.data[24] < 0.0f) { result_bg = fast_math::fminf(result_bg, 255.0f - data[local_row + 4][local_col + 4]); }
				guarded_write(dest_array, idx.global, fast_math::fminf(result_fg, result_bg));
			});
		}
		else
		{
			int kernel_cols_down = ROUNDDOWN(kernel.cols, 4);
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[max_1d_tile_static_size];
				int tile_static_rows = tile_size + kernel.rows & ~1;
				int tile_static_cols = tile_size + kernel.cols & ~1;
				int tile_static_size = tile_static_rows * tile_static_cols;
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - kernel.rows / 2;
				int dy = col - local_col - kernel.cols / 2;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size / 2 - 1);
				int dr = id / tile_static_cols;
				int dc = id % tile_static_cols;
				data[id] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), FLT_MAX);
				data[id + tile_static_size / 2] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_rows / 2, dy + dc), FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();

				float result_fg = FLT_MAX;
				float result_bg = FLT_MAX;
				for(int i = 0; i < kernel.rows; i++)
				{
					int j = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
					for(; j < kernel_cols_down; j += 4)
					{
						float kernel_value = kernel.data[i * kernel.cols + j];
						float data_value = data[(local_row + i) * tile_static_cols + local_col + j];
						if(kernel_value > 0.0f)
						{
							result_fg = fast_math::fminf(result_fg, data_value);
						}
						else if(kernel_value < 0.0f)
						{
							result_bg = fast_math::fminf(result_bg, 255.0f - data_value);
						}
						kernel_value = kernel.data[i * kernel.cols + j + 1];
						data_value = data[(local_row + i) * tile_static_cols + local_col + j + 1];
						if(kernel_value > 0.0f)
						{
							result_fg = fast_math::fminf(result_fg, data_value);
						}
						else if(kernel_value < 0.0f)
						{
							result_bg = fast_math::fminf(result_bg, 255.0f - data_value);
						}
						kernel_value = kernel.data[i * kernel.cols + j + 2];
						data_value = data[(local_row + i) * tile_static_cols + local_col + j + 2];
						if(kernel_value > 0.0f)
						{
							result_fg = fast_math::fminf(result_fg, data_value);
						}
						else if(kernel_value < 0.0f)
						{
							result_bg = fast_math::fminf(result_bg, 255.0f - data_value);
						}
						kernel_value = kernel.data[i * kernel.cols + j + 3];
						data_value = data[(local_row + i) * tile_static_cols + local_col + j + 3];
						if(kernel_value > 0.0f)
						{
							result_fg = fast_math::fminf(result_fg, data_value);
						}
						else if(kernel_value < 0.0f)
						{
							result_bg = fast_math::fminf(result_bg, 255.0f - data_value);
						}
					}
#endif
					for(; j < kernel.cols; j++)
					{
						float kernel_value = kernel.data[i * kernel.cols + j];
						float data_value = data[(local_row + i) * tile_static_cols + local_col + j];
						if(kernel_value > 0.0f)
						{
							result_fg = fast_math::fminf(result_fg, data_value);
						}
						else if(kernel_value < 0.0f)
						{
							result_bg = fast_math::fminf(result_bg, 255.0f - data_value);
						}
					}
				}
				guarded_write(dest_array, idx.global, fast_math::fminf(result_fg, result_bg));
			});
		}
	}
}
