#pragma once

#include "amp_core.h"

namespace amp
{
	// Custom 2D Filtering using supplied functor
	// Supported neighbourhood size: up to 89X89 or 1X223
#if !OPTIMIZE_FOR_INTEL
	template<typename FilterOp>
	inline void filter2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, float border_value, FilterOp filter_op)
	{
		static const int tile_size = 32;
		static const int kernel_cols = FilterOp::cols;
		static const int kernel_rows = FilterOp::rows;
		static const int cache_cols = tile_size + (kernel_cols & ~1);
		static const int cache_rows = tile_size + (kernel_rows & ~1);
		static const int cache_size = cache_cols * cache_rows;
		static const int cache_fetch_rows_per_round = tile_size * tile_size / cache_cols;
		static const int cache_fetch_rounds = (cache_size + cache_fetch_rows_per_round - 1) / cache_fetch_rows_per_round;
		static_assert(cache_size <= 8192, "neighbourhood size is too large");
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float cache[cache_rows][cache_cols];
			int row = idx.global[0];
			int col = idx.global[1];
			int local_row = idx.local[0];
			int local_col = idx.local[1];
			int dy = row - local_row - kernel_rows / 2;
			int dx = col - local_col - kernel_cols / 2;
			const int id = direct3d::imin(local_row * tile_size + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr = id / cache_cols;
			int dc = id % cache_cols;
			for (; dr < cache_rows; dr += cache_fetch_rows_per_round)
			{
				cache[dr][dc] = guarded_read(src_array, concurrency::index<2>(dy + dr, dx + dc), border_value);
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			guarded_write(dest_array, idx.global, filter_op(cache, local_row + kernel_rows / 2, local_col + kernel_cols / 2));
		});
	}

	template<typename FilterOp>
	inline void filter2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, FilterOp filter_op)
	{
		static const int tile_size = 32;
		static const int kernel_cols = FilterOp::cols;
		static const int kernel_rows = FilterOp::rows;
		static const int cache_cols = tile_size + (kernel_cols & ~1);
		static const int cache_rows = tile_size + (kernel_rows & ~1);
		static const int cache_size = cache_cols * cache_rows;
		static const int cache_fetch_rows_per_round = tile_size * tile_size / cache_cols;
		static const int cache_fetch_rounds = (cache_size + cache_fetch_rows_per_round - 1) / cache_fetch_rows_per_round;
		static_assert(cache_size <= 8192, "neighbourhood size is too large");
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float cache[cache_rows][cache_cols];
			int row = idx.global[0];
			int col = idx.global[1];
			int local_row = idx.local[0];
			int local_col = idx.local[1];
			int dy = row - local_row - kernel_rows / 2;
			int dx = col - local_col - kernel_cols / 2;
			const int id = direct3d::imin(local_row * tile_size + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr = id / cache_cols;
			int dc = id % cache_cols;
			for (; dr < cache_rows; dr += cache_fetch_rows_per_round)
			{
				cache[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dy + dr, dx + dc));
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			guarded_write(dest_array, idx.global, filter_op(cache, local_row + kernel_rows / 2, local_col + kernel_cols / 2));
		});
	}
#else
	template<typename FilterOp>
	inline void filter2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, float border_value, FilterOp filter_op)
	{
		static const int tile_size_x = 32;
		static const int tile_size_y = 8;
		static const int tile_div_y = 4;
		static const int kernel_cols = FilterOp::cols;
		static const int kernel_rows = FilterOp::rows;
		static const int cache_cols = tile_size_x + (kernel_cols & ~1);
		static const int cache_rows = tile_size_y * tile_div_y + (kernel_rows & ~1);
		static const int cache_size = cache_cols * cache_rows;
		static const int cache_fetch_rows_per_round = tile_size_x * tile_size_y * tile_div_y / cache_cols;
		static const int cache_fetch_rounds = (cache_size + cache_fetch_rows_per_round - 1) / cache_fetch_rows_per_round;
		static_assert(cache_size <= 8192, "neighbourhood size is too large");
		concurrency::extent<2> ext(DIVUP(src_array.get_extent()[0], tile_div_y), src_array.get_extent()[1]);
		parallel_for_each(acc_view, ext.tile<tile_size_y, tile_size_x>().pad(), [=](tiled_index<tile_size_y, tile_size_x> idx) restrict(amp)
		{
			tile_static float cache[cache_rows][cache_cols];
			int row = idx.global[0] * tile_div_y;
			int col = idx.global[1];
			int local_row = idx.local[0] * tile_div_y;
			int local_col = idx.local[1];
			int dy = row - local_row - kernel_rows / 2;
			int dx = col - local_col - kernel_cols / 2;
			const int id0 = direct3d::imin(local_row * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr0 = id0 / cache_cols;
			int dc0 = id0 % cache_cols;
			for (; dr0 < cache_rows; dr0 += cache_fetch_rows_per_round)
			{
				cache[dr0][dc0] = guarded_read(src_array, concurrency::index<2>(dy + dr0, dx + dc0), border_value);
			}
			const int id1 = direct3d::imin((local_row + 1) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr1 = id1 / cache_cols;
			int dc1 = id1 % cache_cols;
			for (; dr1 < cache_rows; dr1 += cache_fetch_rows_per_round)
			{
				cache[dr1][dc1] = guarded_read(src_array, concurrency::index<2>(dy + dr1, dx + dc1), border_value);
			}
			const int id2 = direct3d::imin((local_row + 2) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr2 = id2 / cache_cols;
			int dc2 = id2 % cache_cols;
			for (; dr2 < cache_rows; dr2 += cache_fetch_rows_per_round)
			{
				cache[dr2][dc2] = guarded_read(src_array, concurrency::index<2>(dy + dr2, dx + dc2), border_value);
			}
			const int id3 = direct3d::imin((local_row + 3) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr3 = id3 / cache_cols;
			int dc3 = id3 % cache_cols;
			for (; dr3 < cache_rows; dr3 += cache_fetch_rows_per_round)
			{
				cache[dr3][dc3] = guarded_read(src_array, concurrency::index<2>(dy + dr3, dx + dc3), border_value);
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			guarded_write(dest_array, concurrency::index<2>(row, col), filter_op(cache, local_row + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 1, col), filter_op(cache, local_row + 1 + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 2, col), filter_op(cache, local_row + 2 + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 3, col), filter_op(cache, local_row + 3 + kernel_rows / 2, local_col + kernel_cols / 2));
		});
	}

	template<typename FilterOp>
	inline void filter2d_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, FilterOp filter_op)
	{
		static const int tile_size_x = 32;
		static const int tile_size_y = 8;
		static const int tile_div_y = 4;
		static const int kernel_cols = FilterOp::cols;
		static const int kernel_rows = FilterOp::rows;
		static const int cache_cols = tile_size_x + (kernel_cols & ~1);
		static const int cache_rows = tile_size_y * tile_div_y + (kernel_rows & ~1);
		static const int cache_size = cache_cols * cache_rows;
		static const int cache_fetch_rows_per_round = tile_size_x * tile_size_y * tile_div_y / cache_cols;
		static const int cache_fetch_rounds = (cache_size + cache_fetch_rows_per_round - 1) / cache_fetch_rows_per_round;
		static_assert(cache_size <= 8192, "neighbourhood size is too large");
		concurrency::extent<2> ext(DIVUP(src_array.get_extent()[0], tile_div_y), src_array.get_extent()[1]);
		parallel_for_each(acc_view, ext.tile<tile_size_y, tile_size_x>().pad(), [=](tiled_index<tile_size_y, tile_size_x> idx) restrict(amp)
		{
			tile_static float cache[cache_rows][cache_cols];
			int row = idx.global[0] * tile_div_y;
			int col = idx.global[1];
			int local_row = idx.local[0] * tile_div_y;
			int local_col = idx.local[1];
			int dy = row - local_row - kernel_rows / 2;
			int dx = col - local_col - kernel_cols / 2;
			const int id0 = direct3d::imin(local_row * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr0 = id0 / cache_cols;
			int dc0 = id0 % cache_cols;
			for (; dr0 < cache_rows; dr0 += cache_fetch_rows_per_round)
			{
				cache[dr0][dc0] = guarded_read_reflect101(src_array, concurrency::index<2>(dy + dr0, dx + dc0));
			}
			const int id1 = direct3d::imin((local_row + 1) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr1 = id1 / cache_cols;
			int dc1 = id1 % cache_cols;
			for (; dr1 < cache_rows; dr1 += cache_fetch_rows_per_round)
			{
				cache[dr1][dc1] = guarded_read_reflect101(src_array, concurrency::index<2>(dy + dr1, dx + dc1));
			}
			const int id2 = direct3d::imin((local_row + 2) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr2 = id2 / cache_cols;
			int dc2 = id2 % cache_cols;
			for (; dr2 < cache_rows; dr2 += cache_fetch_rows_per_round)
			{
				cache[dr2][dc2] = guarded_read_reflect101(src_array, concurrency::index<2>(dy + dr2, dx + dc2));
			}
			const int id3 = direct3d::imin((local_row + 3) * tile_size_x + local_col, cache_fetch_rows_per_round * cache_cols - 1);
			int dr3 = id3 / cache_cols;
			int dc3 = id3 % cache_cols;
			for (; dr3 < cache_rows; dr3 += cache_fetch_rows_per_round)
			{
				cache[dr3][dc3] = guarded_read_reflect101(src_array, concurrency::index<2>(dy + dr3, dx + dc3));
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			guarded_write(dest_array, concurrency::index<2>(row, col), filter_op(cache, local_row + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 1, col), filter_op(cache, local_row + 1 + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 2, col), filter_op(cache, local_row + 2 + kernel_rows / 2, local_col + kernel_cols / 2));
			guarded_write(dest_array, concurrency::index<2>(row + 3, col), filter_op(cache, local_row + 3 + kernel_rows / 2, local_col + kernel_cols / 2));
		});
	}
#endif
}
