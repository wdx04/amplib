#pragma once

#include "amp_core.h"

namespace amp
{
	// Dilate
	template<unsigned int kernel_size>
	inline void dilate_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, const kernel_wrapper<float, kernel_size>& kernel)
	{
		dest_array.discard_data();
		static const int tile_size = 32;
		static const int max_1d_tile_static_size = tile_size * tile_size * 2;
		assert((tile_size + (kernel.cols & ~1)) * (tile_size + (kernel.rows & ~1)) <= max_1d_tile_static_size);
		static const int tile_static_size_3x3 = tile_size + 2;
		static const int tile_static_size_5x5 = tile_size + 4;
		if(!kernel.is_all_positive())
		{
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
					data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[dr + tile_static_size_3x3 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					float result = -FLT_MAX;
					result = fast_math::fmaxf(result, kernel.data[0] > 0.0f ? data[local_row + 0][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[1] > 0.0f ? data[local_row + 0][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[2] > 0.0f ? data[local_row + 0][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[3] > 0.0f ? data[local_row + 1][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[4] > 0.0f ? data[local_row + 1][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[5] > 0.0f ? data[local_row + 1][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[6] > 0.0f ? data[local_row + 2][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[7] > 0.0f ? data[local_row + 2][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[8] > 0.0f ? data[local_row + 2][local_col + 2] : -FLT_MAX);
					guarded_write(dest_array, idx.global, result);
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
					data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[dr + tile_static_size_5x5 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_5x5 / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					float result = -FLT_MAX;
					result = fast_math::fmaxf(result, kernel.data[0] > 0.0f ? data[local_row + 0][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[1] > 0.0f ? data[local_row + 0][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[2] > 0.0f ? data[local_row + 0][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[3] > 0.0f ? data[local_row + 0][local_col + 3] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[4] > 0.0f ? data[local_row + 0][local_col + 4] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[5] > 0.0f ? data[local_row + 1][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[6] > 0.0f ? data[local_row + 1][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[7] > 0.0f ? data[local_row + 1][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[8] > 0.0f ? data[local_row + 1][local_col + 3] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[9] > 0.0f ? data[local_row + 1][local_col + 4] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[10] > 0.0f ? data[local_row + 2][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[11] > 0.0f ? data[local_row + 2][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[12] > 0.0f ? data[local_row + 2][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[13] > 0.0f ? data[local_row + 2][local_col + 3] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[14] > 0.0f ? data[local_row + 2][local_col + 4] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[15] > 0.0f ? data[local_row + 3][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[16] > 0.0f ? data[local_row + 3][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[17] > 0.0f ? data[local_row + 3][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[18] > 0.0f ? data[local_row + 3][local_col + 3] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[19] > 0.0f ? data[local_row + 3][local_col + 4] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[20] > 0.0f ? data[local_row + 4][local_col + 0] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[21] > 0.0f ? data[local_row + 4][local_col + 1] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[22] > 0.0f ? data[local_row + 4][local_col + 2] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[23] > 0.0f ? data[local_row + 4][local_col + 3] : -FLT_MAX);
					result = fast_math::fmaxf(result, kernel.data[24] > 0.0f ? data[local_row + 4][local_col + 4] : -FLT_MAX);
					guarded_write(dest_array, idx.global, result);
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
					data[id] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[id + tile_static_size / 2] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_rows / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					float result = -FLT_MAX;
					for(int i = 0; i < kernel.rows; i++)
					{
						int j = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
						for(; j < kernel_cols_down; j += 4)
						{
							float kernel_value = kernel.data[i * kernel.cols + j];
							float data_value = data[(local_row + i) * tile_static_cols + local_col + j];
							result = fast_math::fmaxf(result, kernel_value > 0.0f ? data_value : -FLT_MAX);
							kernel_value = kernel.data[i * kernel.cols + j + 1];
							data_value = data[(local_row + i) * tile_static_cols + local_col + j + 1];
							result = fast_math::fmaxf(result, kernel_value > 0.0f ? data_value : -FLT_MAX);
							kernel_value = kernel.data[i * kernel.cols + j + 2];
							data_value = data[(local_row + i) * tile_static_cols + local_col + j + 2];
							result = fast_math::fmaxf(result, kernel_value > 0.0f ? data_value : -FLT_MAX);
							kernel_value = kernel.data[i * kernel.cols + j + 3];
							data_value = data[(local_row + i) * tile_static_cols + local_col + j + 3];
							result = fast_math::fmaxf(result, kernel_value > 0.0f ? data_value : -FLT_MAX);
						}
#endif
						for(; j < kernel.cols; j++)
						{
							float kernel_value = kernel.data[i * kernel.cols + j];
							float data_value = data[(local_row + i) * tile_static_cols + local_col + j];
							result = fast_math::fmaxf(result, kernel_value > 0.0f ? data_value : -FLT_MAX);
						}
					}
					guarded_write(dest_array, idx.global, result);
				});
			}
		}
		else
		{
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
					data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[dr + tile_static_size_3x3 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					float result = -FLT_MAX;
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 2]);
					guarded_write(dest_array, idx.global, result);
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
					data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[dr + tile_static_size_5x5 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_5x5 / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					float result = -FLT_MAX;
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 3]);
					result = fast_math::fmaxf(result, data[local_row + 0][local_col + 4]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 3]);
					result = fast_math::fmaxf(result, data[local_row + 1][local_col + 4]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 3]);
					result = fast_math::fmaxf(result, data[local_row + 2][local_col + 4]);
					result = fast_math::fmaxf(result, data[local_row + 3][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 3][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 3][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 3][local_col + 3]);
					result = fast_math::fmaxf(result, data[local_row + 3][local_col + 4]);
					result = fast_math::fmaxf(result, data[local_row + 4][local_col + 0]);
					result = fast_math::fmaxf(result, data[local_row + 4][local_col + 1]);
					result = fast_math::fmaxf(result, data[local_row + 4][local_col + 2]);
					result = fast_math::fmaxf(result, data[local_row + 4][local_col + 3]);
					result = fast_math::fmaxf(result, data[local_row + 4][local_col + 4]);
					guarded_write(dest_array, idx.global, result);
				});
			}
			else
			{
				int kernel_cols_down = ROUNDDOWN(kernel.cols, 4);
				int kernel_rows_down = ROUNDDOWN(kernel.rows, 4);
				parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					tile_static float data[max_1d_tile_static_size];
					tile_static float row_cache[max_1d_tile_static_size];
					const int tile_static_rows = tile_size + kernel.rows & ~1;
					const int tile_static_cols = tile_size + kernel.cols & ~1;
					const int tile_static_size = tile_static_rows * tile_static_cols;
					const int row = idx.global[0];
					const int col = idx.global[1];
					const int local_row = idx.local[0];
					const int local_col = idx.local[1];
					const int dx = row - local_row - kernel.rows / 2;
					const int dy = col - local_col - kernel.cols / 2;
					int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size / 2 - 1);
					int dr = id / tile_static_cols;
					int dc = id % tile_static_cols;
					data[id] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
					data[id + tile_static_size / 2] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_rows / 2, dy + dc), -FLT_MAX);
					idx.barrier.wait_with_tile_static_memory_fence();

					// row pass 1
					float result = -FLT_MAX;
					id = direct3d::imin(local_row * tile_size + local_col, tile_size * tile_static_rows / 2 - 1);
					dr = id / tile_size;
					dc = id % tile_size;
					int i = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
					for (; i < kernel_cols_down; i += 4)
					{
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 1]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 2]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 3]);
					}
#endif
					for (; i < kernel.cols; i++)
					{
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i]);
					}
					row_cache[dr * tile_static_cols + dc + kernel.cols / 2] = result;

					// row pass 2
					result = -FLT_MAX;
					dr += tile_static_rows / 2;
					i = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
					for (; i < kernel_cols_down; i += 4)
					{
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 1]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 2]);
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i + 3]);
					}
#endif
					for (; i < kernel.cols; i++)
					{
						result = fast_math::fmaxf(result, data[dr * tile_static_cols + dc + i]);
					}
					row_cache[dr * tile_static_cols + dc + kernel.cols / 2] = result;
					idx.barrier.wait_with_tile_static_memory_fence();

					// col pass
					int row_in_cache = local_row + kernel.rows / 2;
					int col_in_cache = local_col + kernel.cols / 2;
					int j = 0;
					result = -FLT_MAX;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
					for (; j < kernel_rows_down; j += 4)
					{
						result = fast_math::fmaxf(result, row_cache[(local_row + j) * tile_static_cols + col_in_cache]);
						result = fast_math::fmaxf(result, row_cache[(local_row + j + 1) * tile_static_cols + col_in_cache]);
						result = fast_math::fmaxf(result, row_cache[(local_row + j + 2) * tile_static_cols + col_in_cache]);
						result = fast_math::fmaxf(result, row_cache[(local_row + j + 3) * tile_static_cols + col_in_cache]);
					}
#endif
					for (; j < kernel.rows; j++)
					{
						result = fast_math::fmaxf(result, row_cache[(local_row + j) * tile_static_cols + col_in_cache]);
					}

					// write result
					guarded_write(dest_array, idx.global, result);
				});
			}
		}
	}

	inline void dilate_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array
		, const cv::Mat& kernel, int iterations = 1)
	{
		kernel_wrapper<float, 169U> wrapped_kernel(kernel);
		if(iterations <= 1)
		{
			dilate_32f_c1(acc_view, src_array, dest_array, wrapped_kernel);
		}
		else
		{
			if(iterations % 2 == 0)
			{
				concurrency::copy_async(src_array, dest_array);
				while(iterations > 0)
				{
					dilate_32f_c1(acc_view, dest_array, temp_array, wrapped_kernel);
					dilate_32f_c1(acc_view, temp_array, dest_array, wrapped_kernel);
					iterations -= 2;
				}
			}
			else
			{
				dilate_32f_c1(acc_view, src_array, dest_array, wrapped_kernel);
				while(--iterations > 0)
				{
					dilate_32f_c1(acc_view, dest_array, temp_array, wrapped_kernel);
					dilate_32f_c1(acc_view, temp_array, dest_array, wrapped_kernel);
					--iterations;
				}
			}
		}
	}

}
