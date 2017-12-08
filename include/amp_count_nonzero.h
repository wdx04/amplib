#pragma once

#include "amp_core.h"

namespace amp
{
	// Count non-zero value
	inline int count_nonzero_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array)
	{
		int rows = src_array.get_extent()[0];
		int cols = src_array.get_extent()[1];
		concurrency::array<int, 1> gpu_count(1, acc_view);
		concurrency::parallel_for_each(acc_view, gpu_count.get_extent(), [&gpu_count](concurrency::index<1> idx) restrict(amp)
		{
			gpu_count(idx) = 0;
		});
		static const int tile_size = 64;
		concurrency::parallel_for_each(acc_view, concurrency::extent<2>(rows, tile_size).tile<1, tile_size>().pad(), [=, &gpu_count](concurrency::tiled_index<1, tile_size> idx) restrict(amp)
		{
			tile_static int partial_count;
			int row = idx.global[0];
			int lid = idx.local[1];
			if (lid == 0) partial_count = 0;
			idx.barrier.wait_with_tile_static_memory_fence();

			int local_count = 0;
			for (int col = lid; col < cols; col += tile_size)
			{
				if (src_array(row, col) != 0.0f) local_count++;
			}
			concurrency::atomic_fetch_add(&partial_count, local_count);
			idx.barrier.wait_with_tile_static_memory_fence();

			if (lid == 0) concurrency::atomic_fetch_add(&gpu_count(0), partial_count);
		});
		int nonzero_count = 0;
		concurrency::copy(gpu_count, &nonzero_count);
		return nonzero_count;
	}
}
