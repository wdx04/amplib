#pragma once

#include "amp_core.h"

namespace amp
{
	namespace detail
	{
		inline void thinning_32f_c1_iter_0(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = tile_size + 2;
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size_3x3][tile_static_size_3x3];
				tile_static int mod_flag;
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
					float p2 = data[local_row][local_col + 1];
					float p3 = data[local_row][local_col + 2];
					float p4 = data[local_row + 1][local_col + 2];
					float p5 = data[local_row + 2][local_col + 2];
					float p6 = data[local_row + 2][local_col + 1];
					float p7 = data[local_row + 2][local_col];
					float p8 = data[local_row + 1][local_col];
					float p9 = data[local_row][local_col];

					int A = (p2 == 0.0f && p3 > 0.0f) + (p3 == 0.0f && p4 > 0.0f) +
						(p4 == 0.0f && p5 > 0.0f) + (p5 == 0.0f && p6 > 0.0f) +
						(p6 == 0.0f && p7 > 0.0f) + (p7 == 0.0f && p8 > 0.0f) +
						(p8 == 0.0f && p9 > 0.0f) + (p9 == 0.0f && p2 > 0.0f);
					int B = (p2 > 0.0f) + (p3 > 0.0f) + (p4 > 0.0f) + (p5 > 0.0f)
						+ (p6 > 0.0f) + (p7 > 0.0f) + (p8 > 0.0f) + (p9 > 0.0f);
					float m1 = p2 * p4 * p6;
					float m2 = p4 * p6 * p8;

					bool set_zero = (A == 1 && (B >= 2 && B <= 6) && m1 == 0.0f && m2 == 0.0f);
					dest_array(idx.global) = set_zero ? 0.0f : data[local_row + 1][local_col + 1];
				}
			});
		}

		inline bool thinning_32f_c1_iter_1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, concurrency::array<int, 2>& gpu_mod_flag)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = tile_size + 2;
			static const int bank_count = 8;
			std::vector<int> cpu_mod_flag(gpu_mod_flag.get_extent()[0] * gpu_mod_flag.get_extent()[1]);
			copy(&cpu_mod_flag[0], gpu_mod_flag);
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=, &gpu_mod_flag](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size_3x3][tile_static_size_3x3];
				tile_static int mod_flag[bank_count];
				int row = idx.global[0];
				int col = idx.global[1];
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int dx = row - local_row - 1;
				int dy = col - local_col - 1;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_3x3 * tile_static_size_3x3 / 2 - 1);
				const int bank_id = id % bank_count;
				int dr = id / tile_static_size_3x3;
				int dc = id % tile_static_size_3x3;
				if(id < bank_count) mod_flag[id] = 0;
				data[dr][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), FLT_MAX);
				data[dr + tile_static_size_3x3 / 2][dc] = guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();

				if(dest_array.get_extent().contains(idx.global))
				{
					float p2 = data[local_row][local_col + 1];
					float p3 = data[local_row][local_col + 2];
					float p4 = data[local_row + 1][local_col + 2];
					float p5 = data[local_row + 2][local_col + 2];
					float p6 = data[local_row + 2][local_col + 1];
					float p7 = data[local_row + 2][local_col];
					float p8 = data[local_row + 1][local_col];
					float p9 = data[local_row][local_col];

					int A = (p2 == 0.0f && p3 > 0.0f) + (p3 == 0.0f && p4 > 0.0f) +
						(p4 == 0.0f && p5 > 0.0f) + (p5 == 0.0f && p6 > 0.0f) +
						(p6 == 0.0f && p7 > 0.0f) + (p7 == 0.0f && p8 > 0.0f) +
						(p8 == 0.0f && p9 > 0.0f) + (p9 == 0.0f && p2 > 0.0f);
					int B = (p2 > 0.0f) + (p3 > 0.0f) + (p4 > 0.0f) + (p5 > 0.0f)
						+ (p6 > 0.0f) + (p7 > 0.0f) + (p8 > 0.0f) + (p9 > 0.0f);
					float m1 = p2 * p4 * p8;
					float m2 = p2 * p6 * p8;

					bool set_zero = (A == 1 && (B >= 2 && B <= 6) && m1 == 0.0f && m2 == 0.0f);
					dest_array(idx.global) = set_zero ? 0.0f : data[local_row + 1][local_col + 1];
					if(set_zero && 0.0f != data[local_row + 1][local_col + 1])
					{
						concurrency::atomic_fetch_inc(&mod_flag[bank_id]);
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				if(id == 0)
				{
					gpu_mod_flag(idx.tile) = mod_flag[0] + mod_flag[1] + mod_flag[2] + mod_flag[3]
						+ mod_flag[4] + mod_flag[5] + mod_flag[6] + mod_flag[7];
				}
			});
			copy(gpu_mod_flag, &cpu_mod_flag[0]);
			return std::accumulate(cpu_mod_flag.begin(), cpu_mod_flag.end(), 0) != 0;
		}
	}

	inline void thinning_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, int max_iter = 0)
	{
		dest_array.discard_data();
		concurrency::array<int, 2> gpu_mod_flag(DIVUP(dest_array.get_extent()[0], 32), DIVUP(dest_array.get_extent()[1], 32), acc_view);
		detail::thinning_32f_c1_iter_0(acc_view, src_array, temp_array);
		if(detail::thinning_32f_c1_iter_1(acc_view, temp_array, dest_array, gpu_mod_flag))
		{
			for(int i = 1; i != max_iter; i++)
			{
				detail::thinning_32f_c1_iter_0(acc_view, dest_array, temp_array);
				if(!detail::thinning_32f_c1_iter_1(acc_view, temp_array, dest_array, gpu_mod_flag))
				{
					break;
				}
			}
		}
	}
}
