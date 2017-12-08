#pragma once

#include "amp_core.h"
#include "amp_geodesic_dilate.h"

namespace amp
{
	namespace detail
	{
		inline void pruning_thinning_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = tile_size + 2;
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
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

				if(dest_array.get_extent().contains(idx.global))
				{
					float p1 = data[local_row + 1][local_col + 1];
					float p2 = data[local_row][local_col + 1];
					float p3 = data[local_row][local_col + 2];
					float p4 = data[local_row + 1][local_col + 2];
					float p5 = data[local_row + 2][local_col + 2];
					float p6 = data[local_row + 2][local_col + 1];
					float p7 = data[local_row + 2][local_col];
					float p8 = data[local_row + 1][local_col];
					float p9 = data[local_row][local_col];

					bool b1 = (p1 > 0.0f && p2 > 0.0f && p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b2 = (p1 > 0.0f && p4 > 0.0f && p2 + p6 + p7 + p8 + p9 == 0.0f);
					bool b3 = (p1 > 0.0f && p6 > 0.0f && p2 + p3 + p4 + p8 + p9 == 0.0f);
					bool b4 = (p1 > 0.0f && p8 > 0.0f && p2 + p3 + p4 + p5 + p6 == 0.0f);
					bool b5 = (p1 > 0.0f && p9 > 0.0f && p2 + p3 + p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b6 = (p1 > 0.0f && p3 > 0.0f && p2 + p9 + p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b7 = (p1 > 0.0f && p5 > 0.0f && p2 + p3 + p4 + p9 + p6 + p7 + p8 == 0.0f);
					bool b8 = (p1 > 0.0f && p7 > 0.0f && p2 + p3 + p4 + p5 + p6 + p9 + p8 == 0.0f);

					bool set_zero = b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8;
					dest_array(idx.global) = set_zero ? 0.0f : data[local_row + 1][local_col + 1];
				}
			});
		}

		inline void pruning_hit_or_miss_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = tile_size + 2;
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
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

				if(dest_array.get_extent().contains(idx.global))
				{
					float p1 = data[local_row + 1][local_col + 1];
					float p2 = data[local_row][local_col + 1];
					float p3 = data[local_row][local_col + 2];
					float p4 = data[local_row + 1][local_col + 2];
					float p5 = data[local_row + 2][local_col + 2];
					float p6 = data[local_row + 2][local_col + 1];
					float p7 = data[local_row + 2][local_col];
					float p8 = data[local_row + 1][local_col];
					float p9 = data[local_row][local_col];

					bool b1 = (p1 > 0.0f && p2 > 0.0f && p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b2 = (p1 > 0.0f && p4 > 0.0f && p2 + p6 + p7 + p8 + p9 == 0.0f);
					bool b3 = (p1 > 0.0f && p6 > 0.0f && p2 + p3 + p4 + p8 + p9 == 0.0f);
					bool b4 = (p1 > 0.0f && p8 > 0.0f && p2 + p3 + p4 + p5 + p6 == 0.0f);
					bool b5 = (p1 > 0.0f && p9 > 0.0f && p2 + p3 + p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b6 = (p1 > 0.0f && p3 > 0.0f && p2 + p9 + p4 + p5 + p6 + p7 + p8 == 0.0f);
					bool b7 = (p1 > 0.0f && p5 > 0.0f && p2 + p3 + p4 + p9 + p6 + p7 + p8 == 0.0f);
					bool b8 = (p1 > 0.0f && p7 > 0.0f && p2 + p3 + p4 + p5 + p6 + p9 + p8 == 0.0f);

					bool is_hit = b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8;
					dest_array(idx.global) = is_hit ? 255.0f : 0.0f;
				}
			});
		}
	}

	inline void pruning_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array1, array_view<float, 2> temp_array2, int max_thinning_iter, int max_dilate_iter = 0)
	{
		// 1.thinning
		if((max_thinning_iter % 2) == 1)
		{
			detail::pruning_thinning_32f_c1(acc_view, src_array, dest_array);
			for(int i = 1; i < max_thinning_iter; i += 2)
			{
				detail::pruning_thinning_32f_c1(acc_view, dest_array, temp_array1);
				detail::pruning_thinning_32f_c1(acc_view, temp_array1, dest_array);
			}
		}
		else
		{
			detail::pruning_thinning_32f_c1(acc_view, src_array, temp_array1);
			for(int i = 2; i < max_thinning_iter; i += 2)
			{
				detail::pruning_thinning_32f_c1(acc_view, temp_array1, dest_array);
				detail::pruning_thinning_32f_c1(acc_view, dest_array, temp_array1);
			}
			detail::pruning_thinning_32f_c1(acc_view, temp_array1, dest_array);
		}
		// 2.find endpoints in thinned image
		if(max_dilate_iter == 0)
		{
			max_dilate_iter = max_thinning_iter + 1;
		}
		if(max_dilate_iter > 0)
		{
			detail::pruning_hit_or_miss_32f_c1(acc_view, dest_array, temp_array1);
			// 3.reconstruction on endpoints
			geodesic_dilate_32f_c1(acc_view, temp_array1, temp_array2, src_array, temp_array1, max_dilate_iter);
			// 4.merge results
			static const int tile_size = 32;
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				if(dest_array.get_extent().contains(idx.global))
				{
					dest_array(idx.global) = fast_math::fmaxf(dest_array(idx.global), temp_array2(idx.global));
				}
			});
		}
	}
}
