#pragma once

#include "amp_core.h"

namespace amp
{
	// Fill Holes
	namespace detail
	{
		inline void queued_fill_holes_reconstruct_32f_c1_iter(accelerator_view& acc_view, array_view<float, 2> work_array, array_view<const float, 2> mask_array
			, const concurrency::array<int_2, 1>& seeds, int cpu_seeds_count
			, concurrency::array<int_2, 1>& new_seeds, array_view<int, 1> new_seeds_count)
		{
			static const int tile_size = 64;
			static const int tile_static_size = tile_size * 8;
			static const int loop_count = 100;
			concurrency::extent<1> ext(cpu_seeds_count);
			parallel_for_each(acc_view, ext.tile<tile_size>().pad(), [=, &seeds, &new_seeds](tiled_index<tile_size> idx) restrict(amp)
			{
				tile_static int_2 local_new_seeds[tile_static_size];
				tile_static int new_seed_idx;
				tile_static int next_iter_count;
				tile_static int global_seed_start_idx;
				int lid = idx.local[0];
				int gid = idx.global[0];
				if (gid == 0) new_seeds_count[0] = 0;
				if (lid == 0)
				{
					new_seed_idx = 0;
					next_iter_count = 0;
				}
				idx.barrier.wait_with_all_memory_fence();

				int_2 seed = ext.contains(idx.global) ? seeds(gid) : int_2(-1, -1);
				for (int i = 0; i < loop_count; i++)
				{
					if (seed.x >= 0)
					{
						bool enqueue_up = seed.y > 0 && mask_array(seed.y - 1, seed.x) == 0.0f && work_array(seed.y - 1, seed.x) == 0.0f;
						if (enqueue_up)
						{
							work_array(seed.y - 1, seed.x) = 255.0f;
							local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y - 1);
						}
						bool enqueue_down = seed.y < work_array.get_extent()[0] - 1 && mask_array(seed.y + 1, seed.x) == 0.0f && work_array(seed.y + 1, seed.x) == 0.0f;
						if (enqueue_down)
						{
							work_array(seed.y + 1, seed.x) = 255.0f;
							local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y + 1);
						}
						bool enqueue_left = seed.x > 0 && mask_array(seed.y, seed.x - 1) == 0.0f && work_array(seed.y, seed.x - 1) == 0.0f;
						if (enqueue_left)
						{
							work_array(seed.y, seed.x - 1) = 255.0f;
							local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x - 1, seed.y);
						}
						bool enqueue_right = seed.x < work_array.get_extent()[1] - 1 && mask_array(seed.y, seed.x + 1) == 0.0f && work_array(seed.y, seed.x + 1) == 0.0f;
						if (enqueue_right)
						{
							work_array(seed.y, seed.x + 1) = 255.0f;
							local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x + 1, seed.y);
						}
					}
					idx.barrier.wait_with_all_memory_fence();
					if (lid == 0)
					{
						bool is_next_iter = i != loop_count - 1 && new_seed_idx + 3 * tile_size < tile_static_size;
						if (is_next_iter)
						{
							next_iter_count = new_seed_idx >= tile_size ? tile_size : new_seed_idx;
							new_seed_idx -= next_iter_count;
						}
						else
						{
							next_iter_count = 0;
						}
					}
					idx.barrier.wait_with_tile_static_memory_fence();
					seed = lid < next_iter_count ? local_new_seeds[new_seed_idx + lid] : int_2(-1, -1);
				}
				if (lid == 0)
				{
					global_seed_start_idx = concurrency::atomic_fetch_add(&new_seeds_count[0], new_seed_idx);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
				lid += tile_size;
				if (lid < new_seed_idx) new_seeds(global_seed_start_idx + lid) = local_new_seeds[lid];
			});
		}
	}

	inline void fill_holes_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		accelerator_view cpu_acc_view = concurrency::accelerator(concurrency::accelerator::cpu_accelerator).create_view();
		concurrency::extent<1> ext((dest_array.get_extent()[0] + dest_array.get_extent()[1] - 2) * 2);
		concurrency::array<int_2, 1> seeds1(ext[0] * 16, acc_view);
		concurrency::array<int_2, 1> seeds2(ext[0] * 16, acc_view);
		concurrency::array<int, 1> staging_seeds_count1(1, cpu_acc_view, acc_view);
		array_view<int, 1> seeds_count1(staging_seeds_count1);
		concurrency::array<int, 1> staging_seeds_count2(1, cpu_acc_view, acc_view);
		array_view<int, 1> seeds_count2(staging_seeds_count2);
		seeds_count1[0] = 0;
		seeds_count2[0] = 0;
		// 1.create initial seed queue
		static const int tile_size = 32;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=, &seeds1](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if (dest_array.get_extent().contains(gidx))
			{
				bool blank_edge = (gidx[0] == 0 || gidx[1] == 0 || gidx[0] == dest_array.get_extent()[0] - 1 || gidx[1] == dest_array.get_extent()[1] - 1) && src_array(gidx) == 0.0f;
				if (blank_edge)
				{
					dest_array(gidx) = 255.0f;
					seeds1[concurrency::atomic_fetch_inc(&seeds_count1[0])] = int_2(gidx[1], gidx[0]);
				}
				else
				{
					dest_array(gidx) = 0.0f;
				}
			}
		});
		// 2.reconstruct
		int cpu_seeds_count1 = seeds_count1[0];
		int cpu_seeds_count2 = 0;
		while (true)
		{
			detail::queued_fill_holes_reconstruct_32f_c1_iter(acc_view, dest_array, src_array, seeds1, cpu_seeds_count1, seeds2, seeds_count2);
			if ((cpu_seeds_count2 = seeds_count2[0]) == 0) break;
			if (seeds1.get_extent()[0] < cpu_seeds_count2 * 8)
			{
				seeds1 = concurrency::array<int_2, 1>(std::max(cpu_seeds_count2 * 8, seeds1.get_extent()[0] * 2), acc_view);
			}
			detail::queued_fill_holes_reconstruct_32f_c1_iter(acc_view, dest_array, src_array, seeds2, cpu_seeds_count2, seeds1, seeds_count1);
			if ((cpu_seeds_count1 = seeds_count1[0]) == 0) break;
			if (seeds2.get_extent()[0] < cpu_seeds_count1 * 8)
			{
				seeds2 = concurrency::array<int_2, 1>(std::max(cpu_seeds_count1 * 8, seeds2.get_extent()[0] * 2), acc_view);
			}
		}
		// 3.negate
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if (dest_array.get_extent().contains(idx.global))
			{
				dest_array(idx.global) = 255.0f - dest_array(idx.global);
			}
		});
	}

}
