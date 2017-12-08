#pragma once

#include "amp_core.h"
#include "amp_conv_separable.h"
#include "amp_calc_sat.h"

namespace amp
{
	// Normalized Box Filter
	// On AMD Cards: always use the convolve_separable based version
	// On NVidia/Intel Cards: use the specialized version for kernel size <= 215
	// Note: tile_size may be 32, 64, 128 or 256 depending on size_x and source width
	template<int tile_size>
	static inline void box_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, int size_x, int size_y)
	{
		assert(size_x <= 215);
		assert(size_y <= 215);
		float alpha = 1.0f / float(size_x * size_y);
		int dest_rows = dest_array.get_extent()[0];
		int dest_cols = dest_array.get_extent()[1];
		int block_size_x = tile_size;
		int block_size_y = dest_rows;
		int anchor_x = size_x / 2;
		int anchor_y = size_y / 2;
		int size_x_down = ROUNDDOWN(size_x, 8);
		int size_y_down = ROUNDDOWN(size_y, 8);
		concurrency::extent<2> ext(DIVUP(dest_rows, block_size_y), DIVUP(dest_cols, block_size_x - (size_x - 1)) * block_size_x);
		dest_array.discard_data();
		parallel_for_each(acc_view, ext.tile<1, tile_size>().pad(), [=](concurrency::tiled_index<1, tile_size> idx) restrict(amp)
		{
			float data[256]; // was size_y
			tile_static float col_sum[tile_size];

			int local_id = idx.local[1];
			int x = local_id + (tile_size - (size_x - 1)) * idx.tile[1] - anchor_x;
			int y = idx.global[0] * block_size_y;
			concurrency::index<2> src_pos(y - anchor_y, x);
			concurrency::index<2> dst_pos(y, x);
			float tmp_sum = 0.0f;
			int sy = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
			for(; sy < size_y_down; sy += 8, src_pos[0] += 8)
			{
				data[sy] = guarded_read_reflect101(src_array, src_pos);
				data[sy + 1] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 1, src_pos[1]));
				data[sy + 2] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 2, src_pos[1]));
				data[sy + 3] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 3, src_pos[1]));
				data[sy + 4] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 4, src_pos[1]));
				data[sy + 5] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 5, src_pos[1]));
				data[sy + 6] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 6, src_pos[1]));
				data[sy + 7] = guarded_read_reflect101(src_array, concurrency::index<2>(src_pos[0] + 7, src_pos[1]));
				tmp_sum += (data[sy] + data[sy + 1] + data[sy + 2] + data[sy + 3] + data[sy + 4] + data[sy + 5] + data[sy + 6] + data[sy + 7]);
			}
#endif
			for(; sy < size_y; sy++, src_pos[0]++)
			{
				tmp_sum += data[sy] = guarded_read_reflect101(src_array, src_pos);
			}
			col_sum[local_id] = tmp_sum;
			idx.barrier.wait_with_tile_static_memory_fence();

			int sy_index = 0; // current index in data[] array
			for(int i = 0; i < block_size_y; ++i)
			{
				if(local_id >= anchor_x && local_id < tile_size - (size_x - 1 - anchor_x) && x >= 0 && x < dest_cols)
				{
					float total_sum = 0.0f;
					int sx = 0;
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
					for(; sx < size_x_down; sx += 8)
					{
						total_sum += (col_sum[local_id + sx - anchor_x]
							+ col_sum[local_id + sx + 1 - anchor_x]
							+ col_sum[local_id + sx + 2 - anchor_x]
							+ col_sum[local_id + sx + 3 - anchor_x]
							+ col_sum[local_id + sx + 4 - anchor_x]
							+ col_sum[local_id + sx + 5 - anchor_x]
							+ col_sum[local_id + sx + 6 - anchor_x]
							+ col_sum[local_id + sx + 7 - anchor_x]);
					}
#endif
					for(; sx < size_x; sx++)
					{
						total_sum += col_sum[local_id + sx - anchor_x];
					}

					guarded_write(dest_array, dst_pos, alpha * total_sum);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				tmp_sum = col_sum[local_id];
				tmp_sum -= data[sy_index];

				data[sy_index] = guarded_read_reflect101(src_array, src_pos);
				src_pos[0]++;

				tmp_sum += data[sy_index];
				col_sum[local_id] = tmp_sum;

				sy_index = sy_index + 1 < size_y ? sy_index + 1 : 0;
				idx.barrier.wait_with_tile_static_memory_fence();

				dst_pos[0]++;
			}
		});
	}

	inline void box_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> row_temp_array, int size_x, int size_y)
	{
		cv::Mat kernel_x(1, size_x, CV_32FC1);
		for(int i = 0; i < size_x; i++)
		{
			kernel_x.at<float>(0, i) = 1.0f / float(size_x);
		}
		cv::Mat kernel_y(1, size_y, CV_32FC1);
		for(int i = 0; i < size_y; i++)
		{
			kernel_y.at<float>(0, i) = 1.0f / float(size_y);
		}
		convolve_separable_32f_c1(acc_view, src_array, dest_array, row_temp_array, kernel_x, kernel_y);
	}

	inline void box_filter_with_sat_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> sat_array, int size_x, int size_y, bool reuse_sat = false)
	{
		if(!reuse_sat)
		{
			calc_sat_32f_c1(acc_view, src_array, sat_array, dest_array);
		}
		static const int tile_size = 32;
		dest_array.discard_data();
		int half_size_x = size_x / 2;
		int half_size_y = size_y / 2;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if(dest_array.get_extent().contains(gidx))
			{
				concurrency::index<2> sat_0(direct3d::imax(gidx[0] - half_size_y - 1, -1), direct3d::imax(gidx[1] - half_size_x - 1, -1));
				concurrency::index<2> sat_1(direct3d::imax(gidx[0] - half_size_y - 1, -1), direct3d::imin(gidx[1] + half_size_x, dest_array.get_extent()[1] - 1));
				concurrency::index<2> sat_2(direct3d::imin(gidx[0] + half_size_y, dest_array.get_extent()[0] - 1), direct3d::imax(gidx[1] - half_size_x - 1, -1));
				concurrency::index<2> sat_3(direct3d::imin(gidx[0] + half_size_y, dest_array.get_extent()[0] - 1), direct3d::imin(gidx[1] + half_size_x, dest_array.get_extent()[1] - 1));
				float sum = guarded_read(sat_array, sat_3) + guarded_read(sat_array, sat_0) - guarded_read(sat_array, sat_1) - guarded_read(sat_array, sat_2);
				float count = float((sat_3[0] - sat_0[0]) * (sat_3[1] - sat_0[1]));
				guarded_write(dest_array, gidx, sum / count);
			}
		});
	}

}
