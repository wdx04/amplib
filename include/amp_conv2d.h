#pragma once

#include "amp_core.h"

namespace amp
{
	inline void convolve2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const kernel_wrapper<float, 169U>& wrapped_kernel)
	{
		assert(wrapped_kernel.rows >= 3 && wrapped_kernel.rows <= 13);
		assert(wrapped_kernel.cols >= 3 && wrapped_kernel.cols <= 13);
		dest_array.discard_data();
		static const int tile_size = 32;
		static const int max_kernel_size = 13;
		static const int tile_static_size = tile_size + (max_kernel_size & ~1);
		if(wrapped_kernel.rows == 3 && wrapped_kernel.cols == 3)
		{
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size][tile_static_size];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - 1;
				int dy = col - local_col - 1;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size * tile_static_size / 2 - 1);
				int dr = id / tile_static_size;
				int dc = id % tile_static_size;
				data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
				data[dr + tile_static_size / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size / 2, dy + dc));
				idx.barrier.wait_with_tile_static_memory_fence();

				float result = 0.0f;
				result = direct3d::mad(wrapped_kernel.data[0], data[local_row + 0][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[1], data[local_row + 0][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[2], data[local_row + 0][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[3], data[local_row + 1][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[4], data[local_row + 1][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[5], data[local_row + 1][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[6], data[local_row + 2][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[7], data[local_row + 2][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[8], data[local_row + 2][local_col + 2], result);
				guarded_write(dest_array, idx.global, result);
			});
		}
		else if(wrapped_kernel.rows == 5 && wrapped_kernel.cols == 5)
		{
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size][tile_static_size];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - 2;
				int dy = col - local_col - 2;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size * tile_static_size / 2 - 1);
				int dr = id / tile_static_size;
				int dc = id % tile_static_size;
				data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
				data[dr + tile_static_size / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size / 2, dy + dc));
				idx.barrier.wait_with_tile_static_memory_fence();

				float result = 0.0f;
				result = direct3d::mad(wrapped_kernel.data[0], data[local_row + 0][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[1], data[local_row + 0][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[2], data[local_row + 0][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[3], data[local_row + 0][local_col + 3], result);
				result = direct3d::mad(wrapped_kernel.data[4], data[local_row + 0][local_col + 4], result);
				result = direct3d::mad(wrapped_kernel.data[5], data[local_row + 1][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[6], data[local_row + 1][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[7], data[local_row + 1][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[8], data[local_row + 1][local_col + 3], result);
				result = direct3d::mad(wrapped_kernel.data[9], data[local_row + 1][local_col + 4], result);
				result = direct3d::mad(wrapped_kernel.data[10], data[local_row + 2][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[11], data[local_row + 2][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[12], data[local_row + 2][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[13], data[local_row + 2][local_col + 3], result);
				result = direct3d::mad(wrapped_kernel.data[14], data[local_row + 2][local_col + 4], result);
				result = direct3d::mad(wrapped_kernel.data[15], data[local_row + 3][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[16], data[local_row + 3][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[17], data[local_row + 3][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[18], data[local_row + 3][local_col + 3], result);
				result = direct3d::mad(wrapped_kernel.data[19], data[local_row + 3][local_col + 4], result);
				result = direct3d::mad(wrapped_kernel.data[20], data[local_row + 4][local_col + 0], result);
				result = direct3d::mad(wrapped_kernel.data[21], data[local_row + 4][local_col + 1], result);
				result = direct3d::mad(wrapped_kernel.data[22], data[local_row + 4][local_col + 2], result);
				result = direct3d::mad(wrapped_kernel.data[23], data[local_row + 4][local_col + 3], result);
				result = direct3d::mad(wrapped_kernel.data[24], data[local_row + 4][local_col + 4], result);
				guarded_write(dest_array, idx.global, result);
			});
		}
		else
		{
			parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size][tile_static_size];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - wrapped_kernel.rows / 2;
				int dy = col - local_col - wrapped_kernel.cols / 2;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size * tile_static_size / 2 - 1);
				int dr = id / tile_static_size;
				int dc = id % tile_static_size;
				data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
				data[dr + tile_static_size / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size / 2, dy + dc));
				idx.barrier.wait_with_tile_static_memory_fence();

				float result = 0.0f;
				for(int i = 0; i < wrapped_kernel.rows; i++)
				{
					for(int j = 0; j < wrapped_kernel.cols; j++)
					{
						result = direct3d::mad(wrapped_kernel.data[i * wrapped_kernel.cols + j], data[local_row + i][local_col + j], result);
					}
				}
				guarded_write(dest_array, idx.global, result);
			});
		}
	}

	inline void convolve2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const cv::Mat& kernel)
	{
		kernel_wrapper<float, 169U> wrapped_kernel(kernel);
		convolve2d_32f_c1(acc_view, src_array, dest_array, wrapped_kernel);
	}

}
