#pragma once

#include "amp_core.h"

namespace amp
{
	// Convolve by row
	template<unsigned int max_kernel_size>
	inline void convolve_by_row_32f_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray, array_view<float, 2> destArray, const kernel_wrapper<float, max_kernel_size>& kernel)
	{
		static const int tile_size = 32;
		parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float tile_part[tile_size][tile_size * 2 + 1];
			int global_row = idx.global[0];
			int global_column_reflected = idx.global[1] - kernel.size / 2;
			int local_row = idx.local[0];
			int local_column = idx.local[1];
			float sum = 0.0f;

			tile_part[local_row][tile_size + local_column] = guarded_read_reflect101_for_row(srcArray, concurrency::index<2>(global_row, global_column_reflected));
			int loop_count = (kernel.size + tile_size - 1) / tile_size - 1;
			for(int l = 0; l < loop_count; l++)
			{
				int kernel_offset = tile_size * l;
				tile_part[local_row][local_column] = tile_part[local_row][local_column + tile_size];
				tile_part[local_row][local_column + tile_size] = guarded_read_reflect101_for_row(srcArray, concurrency::index<2>(global_row, global_column_reflected + tile_size * (l + 1)));
				idx.barrier.wait_with_tile_static_memory_fence();

				for(int i = 0; i < tile_size; i++)
				{
					sum = direct3d::mad(kernel.data[kernel_offset + i], tile_part[local_row][local_column + i], sum);
				}
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			int kernel_offset = tile_size * loop_count;
			tile_part[local_row][local_column] = tile_part[local_row][local_column + tile_size];
			tile_part[local_row][local_column + tile_size] = guarded_read_reflect101_for_row(srcArray, concurrency::index<2>(global_row, global_column_reflected + tile_size * (loop_count + 1)));
			idx.barrier.wait_with_tile_static_memory_fence();

			for(int i = 0; i < (kernel.size % tile_size); i++)
			{
				sum = direct3d::mad(kernel.data[kernel_offset + i], tile_part[local_row][local_column + i], sum);
			}
			guarded_write(destArray, idx.global, sum);
		});
	}

	template<unsigned int max_kernel_size = 1024U>
	inline void convolve2_by_row_32f_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray, array_view<float, 2> destArray, const kernel_wrapper<float, max_kernel_size>& kernel)
	{
		static const int tile_size = 256;
		parallel_for_each(acc_view, destArray.get_extent().tile<1, tile_size>().pad(), [=](tiled_index<1, tile_size> idx) restrict(amp)
		{
			tile_static float tile_part[tile_size + max_kernel_size - 1];
			int global_row = idx.global[0];
			int global_col = idx.global[1];
			int global_start_col = idx.tile_origin[1] - kernel.size / 2;
			int local_col = idx.local[1];
			for (int i = local_col; i < tile_size + kernel.size - 1; i += tile_size)
			{
				tile_part[i] = amp::guarded_read_reflect101_for_row(srcArray, concurrency::index<2>(global_row, global_start_col + i));
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			float sum = 0.0f;
			for (int i = 0; i < kernel.size; i++)
			{
				sum = direct3d::mad(kernel.data[i], tile_part[local_col + i], sum);
			}
			guarded_write(destArray, idx.global, sum);
		});
	}

	// Convolve by column
	template<unsigned int max_kernel_size>
	inline void convolve_by_column_32f_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray, array_view<float, 2> destArray, const kernel_wrapper<float, max_kernel_size>& kernel)
	{
		static const int tile_size = 32;
		parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float tile_part[tile_size * 2][tile_size + 1];
			int global_row_reflected = idx.global[0] - kernel.size / 2;
			int global_column = idx.global[1];
			int local_row = idx.local[0];
			int local_column = idx.local[1];
			float sum = 0.0f;

			tile_part[tile_size + local_row][local_column] = guarded_read_reflect101_for_col(srcArray, concurrency::index<2>(global_row_reflected, global_column));
			int loop_count = (kernel.size + tile_size - 1) / tile_size - 1;
			for(int l = 0; l < loop_count; l++)
			{
				int kernel_offset = tile_size * l;
				tile_part[local_row][local_column] = tile_part[local_row + tile_size][local_column];
				tile_part[local_row + tile_size][local_column] = guarded_read_reflect101_for_col(srcArray, concurrency::index<2>(global_row_reflected + tile_size * (l + 1), global_column));
				idx.barrier.wait_with_tile_static_memory_fence();

				for(int i = 0; i < tile_size; i++)
				{
					sum = direct3d::mad(kernel.data[kernel_offset + i], tile_part[local_row + i][local_column], sum);
				}
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			int kernel_offset = tile_size * loop_count;
			tile_part[local_row][local_column] = tile_part[local_row + tile_size][local_column];
			tile_part[local_row + tile_size][local_column] = guarded_read_reflect101_for_col(srcArray, concurrency::index<2>(global_row_reflected + tile_size * (loop_count + 1), global_column));
			idx.barrier.wait_with_tile_static_memory_fence();

			for(int i = 0; i < (kernel.size % tile_size); i++)
			{
				sum = direct3d::mad(kernel.data[kernel_offset + i], tile_part[local_row + i][local_column], sum);
			}
			guarded_write(destArray, idx.global, sum);
		});
	}

	template<unsigned int max_kernel_size = 1024U>
	inline void convolve_separable_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> row_temp_array, const cv::Mat& row_kernel_mat, const cv::Mat& col_kernel_mat)
	{
		kernel_wrapper<float, max_kernel_size> wrappedRowKernel(row_kernel_mat);
		kernel_wrapper<float, max_kernel_size> wrappedColKernel(col_kernel_mat);
		row_temp_array.discard_data();
		convolve2_by_row_32f_c1<max_kernel_size>(acc_view, src_array, row_temp_array, wrappedRowKernel);
		dest_array.discard_data();
		convolve_by_column_32f_c1<max_kernel_size>(acc_view, row_temp_array, dest_array, wrappedColKernel);
	}

}
