#pragma once

#include "amp_core.h"

namespace amp
{
	// Functors
	class dilate_4_connective
	{
	public:
		static const int connective = 4;

		template<int data_size>
		float operator()(float data[][data_size], int local_row, int local_col) const restrict(amp)
		{
			float result = -FLT_MAX;
			result = fast_math::fmaxf(result, data[local_row + 0][local_col + 1]);
			result = fast_math::fmaxf(result, data[local_row + 1][local_col + 0]);
			result = fast_math::fmaxf(result, data[local_row + 1][local_col + 1]);
			result = fast_math::fmaxf(result, data[local_row + 1][local_col + 2]);
			result = fast_math::fmaxf(result, data[local_row + 2][local_col + 1]);
			return result;
		}
	};

	class dilate_8_connective
	{
	public:
		static const int connective = 8;

		template<int data_size>
		float operator()(float data[][data_size], int local_row, int local_col) const restrict(amp)
		{
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
			return result;
		}
	};

	// Geodesic dilation
	namespace detail
	{
		// Normal version with modification flag
		template<typename MorphOp>
		inline bool geodesic_dilate_32f_c1_iter(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
			, array_view<const float, 2> mask_array, concurrency::array<int, 2>& gpu_mod_flag, MorphOp dilate_op)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = 34;
			static const int bank_count = 8;
			std::vector<int> cpu_mod_flag(gpu_mod_flag.get_extent()[0] * gpu_mod_flag.get_extent()[1]);
			copy_async(&cpu_mod_flag[0], gpu_mod_flag);
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
				if (id < bank_count) mod_flag[id] = 0;
				data[dr][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
				data[dr + tile_static_size_3x3 / 2][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), -FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();
				if (dest_array.get_extent().contains(idx.global))
				{
					float result = dilate_op(data, local_row, local_col);
					float new_value = fast_math::fminf(result, mask_array(idx.global));
					dest_array(idx.global) = new_value;
					if (new_value != data[local_row + 1][local_col + 1])
					{
						concurrency::atomic_fetch_inc(&mod_flag[bank_id]);
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				if (id == 0)
				{
					gpu_mod_flag(idx.tile) = mod_flag[0] + mod_flag[1] + mod_flag[2] + mod_flag[3]
						+ mod_flag[4] + mod_flag[5] + mod_flag[6] + mod_flag[7];
				}
			});
			copy(gpu_mod_flag, &cpu_mod_flag[0]);
			return std::accumulate(cpu_mod_flag.begin(), cpu_mod_flag.end(), 0) != 0;
		}

		// Normal version without modification flag
		template<typename MorphOp>
		inline void geodesic_dilate_32f_c1_iter(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
			, array_view<const float, 2> mask_array, MorphOp dilate_op)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = 34;
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
				data[dr][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
				data[dr + tile_static_size_3x3 / 2][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), -FLT_MAX);
				idx.barrier.wait_with_tile_static_memory_fence();
				if (dest_array.get_extent().contains(idx.global))
				{
					dest_array(idx.global) = fast_math::fminf(dilate_op(data, local_row, local_col), mask_array(idx.global));
				}
			});
		}

		// Generate initial queue
		template<typename MorphOp>
		inline void geodesic_dilate_32f_c1_iter(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
			, array_view<const float, 2> mask_array, concurrency::array<int_2, 1>& seeds, array_view<int, 1> seeds_count, MorphOp dilate_op)
		{
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = 34;
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=, &seeds](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static float data[tile_static_size_3x3][tile_static_size_3x3];
				tile_static int_2 shared_queue[tile_size * tile_size];
				tile_static int global_queue_index;
				tile_static int shared_queue_index;
				int row = idx.global[0];
				int col = idx.global[1];
				if (row == 0 && col == 0) seeds_count[0] = 0;
				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int lid = local_row * tile_size + local_col;
				int dx = row - local_row - 1;
				int dy = col - local_col - 1;
				const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_3x3 * tile_static_size_3x3 / 2 - 1);
				int dr = id / tile_static_size_3x3;
				int dc = id % tile_static_size_3x3;
				data[dr][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr, dy + dc), -FLT_MAX);
				data[dr + tile_static_size_3x3 / 2][dc] = amp::guarded_read(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc), -FLT_MAX);
				if (lid == 0) shared_queue_index = 0;
				idx.barrier.wait_with_all_memory_fence();
				if (dest_array.get_extent().contains(idx.global))
				{
					float old_value = data[local_row + 1][local_col + 1];
					float new_value = fast_math::fminf(dilate_op(data, local_row, local_col), mask_array(idx.global));
					dest_array(idx.global) = new_value;
					if (old_value != new_value)
					{
						int index = concurrency::atomic_fetch_inc(&shared_queue_index);
						shared_queue[index].x = col;
						shared_queue[index].y = row;
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				if (lid == 0)
				{
					global_queue_index = concurrency::atomic_fetch_add(&seeds_count[0], shared_queue_index);
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				if (global_queue_index + shared_queue_index < seeds.get_extent()[0])
				{
					if (lid < shared_queue_index)
					{
						seeds(global_queue_index + lid) = shared_queue[lid];
					}
				}
			});
		}

		// Use input/output queue
		inline void geodesic_dilate_32f_c1_connective4_iter(accelerator_view& acc_view, array_view<float, 2> work_array, array_view<const float, 2> mask_array
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
						if (seed.y > 0)
						{
							float old_value = work_array(seed.y - 1, seed.x);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 2, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y - 1, seed.x));
							if (new_value != old_value)
							{
								work_array(seed.y - 1, seed.x) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y - 1);
							}
						}
						if (seed.y < work_array.get_extent()[0] - 1)
						{
							float old_value = work_array(seed.y + 1, seed.x);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 2, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y + 1, seed.x));
							if (new_value != old_value)
							{
								work_array(seed.y + 1, seed.x) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y + 1);
							}
						}
						if (seed.x > 0)
						{
							float old_value = work_array(seed.y, seed.x - 1);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x - 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y, seed.x - 1));
							if (new_value != old_value)
							{
								work_array(seed.y, seed.x - 1) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x - 1, seed.y);
							}
						}
						if (seed.x < work_array.get_extent()[1] - 1)
						{
							float old_value = work_array(seed.y, seed.x + 1);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x + 2), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y, seed.x + 1));
							if (new_value != old_value)
							{
								work_array(seed.y, seed.x + 1) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x + 1, seed.y);
							}
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

		inline void geodesic_dilate_32f_c1_connective8_iter(accelerator_view& acc_view, array_view<float, 2> work_array, array_view<const float, 2> mask_array
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
						if (seed.y > 0)
						{
							float old_value = work_array(seed.y - 1, seed.x);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 2, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 2, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 2, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y - 1, seed.x));
							if (new_value != old_value)
							{
								work_array(seed.y - 1, seed.x) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y - 1);
							}
						}
						if (seed.y < work_array.get_extent()[0] - 1)
						{
							float old_value = work_array(seed.y + 1, seed.x);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 2, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 2, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 2, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y + 1, seed.x));
							if (new_value != old_value)
							{
								work_array(seed.y + 1, seed.x) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x, seed.y + 1);
							}
						}
						if (seed.x > 0)
						{
							float old_value = work_array(seed.y, seed.x - 1);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x - 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x - 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x - 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x - 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y, seed.x - 1));
							if (new_value != old_value)
							{
								work_array(seed.y, seed.x - 1) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x - 1, seed.y);
							}
						}
						if (seed.x < work_array.get_extent()[1] - 1)
						{
							float old_value = work_array(seed.y, seed.x + 1);
							float new_value = fast_math::fmaxf(old_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x + 1), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y, seed.x + 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y + 1, seed.x + 2), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x), -FLT_MAX));
							new_value = fast_math::fmaxf(new_value, guarded_read(work_array, concurrency::index<2>(seed.y - 1, seed.x + 2), -FLT_MAX));
							new_value = fast_math::fminf(new_value, mask_array(seed.y, seed.x + 1));
							if (new_value != old_value)
							{
								work_array(seed.y, seed.x + 1) = new_value;
								local_new_seeds[concurrency::atomic_fetch_inc(&new_seed_idx)] = int_2(seed.x + 1, seed.y);
							}
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

	template<typename MorphOp>
	inline void geodesic_dilate_32f_c1(accelerator_view acc_view, array_view<float, 2> dest_array, array_view<const float, 2> mask_array
		, array_view<float, 2> temp_array, int max_iter)
	{
		MorphOp dilate_op;
		if (max_iter == 0)
		{
			accelerator_view cpu_acc_view = concurrency::accelerator(concurrency::accelerator::cpu_accelerator).create_view();
			concurrency::extent<1> ext(dest_array.get_extent()[0] * dest_array.get_extent()[1]);
			concurrency::array<int_2, 1> seeds1(ext, acc_view);
			concurrency::array<int_2, 1> seeds2(ext, acc_view);
			concurrency::array<int, 1> staging_seeds_count1(1, cpu_acc_view, acc_view);
			array_view<int, 1> seeds_count1(staging_seeds_count1);
			concurrency::array<int, 1> staging_seeds_count2(1, cpu_acc_view, acc_view);
			array_view<int, 1> seeds_count2(staging_seeds_count2);
			seeds_count1[0] = 0;
			seeds_count2[0] = 0;
			detail::geodesic_dilate_32f_c1_iter(acc_view, dest_array, temp_array, mask_array, dilate_op);
			detail::geodesic_dilate_32f_c1_iter(acc_view, temp_array, dest_array, mask_array, seeds1, seeds_count1, dilate_op);
			int cpu_seeds_count1 = seeds_count1[0];
			int cpu_seeds_count2 = 0;
			if (cpu_seeds_count1 > 0)
			{
				if (MorphOp::connective == 4)
				{
					while (true)
					{
						detail::geodesic_dilate_32f_c1_connective4_iter(acc_view, dest_array, mask_array, seeds1, cpu_seeds_count1, seeds2, seeds_count2);
						if ((cpu_seeds_count2 = seeds_count2[0]) == 0) break;
						detail::geodesic_dilate_32f_c1_connective4_iter(acc_view, dest_array, mask_array, seeds2, cpu_seeds_count2, seeds1, seeds_count1);
						if ((cpu_seeds_count1 = seeds_count1[0]) == 0) break;
					}
				}
				else
				{
					while (true)
					{
						detail::geodesic_dilate_32f_c1_connective8_iter(acc_view, dest_array, mask_array, seeds1, cpu_seeds_count1, seeds2, seeds_count2);
						if ((cpu_seeds_count2 = seeds_count2[0]) == 0) break;
						detail::geodesic_dilate_32f_c1_connective8_iter(acc_view, dest_array, mask_array, seeds2, cpu_seeds_count2, seeds1, seeds_count1);
						if ((cpu_seeds_count1 = seeds_count1[0]) == 0) break;
					}
				}
			}
		}
		else
		{
			concurrency::array<int, 2> gpu_mod_flag(DIVUP(dest_array.get_extent()[0], 32), DIVUP(dest_array.get_extent()[1], 32), acc_view);
			int i = 0;
			while (true)
			{
				detail::geodesic_dilate_32f_c1_iter(acc_view, dest_array, temp_array, mask_array, dilate_op);
				if (++i == max_iter)
				{
					concurrency::copy_async(temp_array, dest_array);
					break;
				}
				if (!detail::geodesic_dilate_32f_c1_iter(acc_view, temp_array, dest_array, mask_array, gpu_mod_flag, dilate_op) || ++i == max_iter)
				{
					break;
				}
			}
		}
	}

	template<typename MorphOp>
	inline void geodesic_dilate_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<const float, 2> mask_array, array_view<float, 2> temp_array, int max_iter)
	{
		concurrency::copy_async(src_array, dest_array);
		geodesic_dilate_32f_c1<MorphOp>(acc_view, dest_array, mask_array, temp_array, max_iter);
	}

	inline void geodesic_dilate_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<const float, 2> mask_array, array_view<float, 2> temp_array, int max_iter = 0, bool connective_4 = false)
	{
		if (connective_4)
		{
			geodesic_dilate_32f_c1<dilate_4_connective>(acc_view, src_array, dest_array, mask_array, temp_array, max_iter);
		}
		else
		{
			geodesic_dilate_32f_c1<dilate_8_connective>(acc_view, src_array, dest_array, mask_array, temp_array, max_iter);
		}
	}

	inline void geodesic_dilate_32f_c1(accelerator_view acc_view, array_view<float, 2> dest_array, array_view<const float, 2> mask_array
		, array_view<float, 2> temp_array, int max_iter = 0, bool connective_4 = false)
	{
		if (connective_4)
		{
			geodesic_dilate_32f_c1<dilate_4_connective>(acc_view, dest_array, mask_array, temp_array, max_iter);
		}
		else
		{
			geodesic_dilate_32f_c1<dilate_8_connective>(acc_view, dest_array, mask_array, temp_array, max_iter);
		}
	}
}
