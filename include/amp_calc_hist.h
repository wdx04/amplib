#pragma once

#include "amp_core.h"

namespace amp
{
	inline void calc_hist_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<int, 1> dest_array, int bin_size = 256, bool retain = false)
	{
#if OPTIMIZE_FOR_AMD || OPTIMIZE_FOR_INTEL
		static const int tile_size = 32;
#else
		static const int tile_size = 16;
#endif
		static const int max_bin_size = 256;
		static const int bank_count = 8;
		const float bin_ratio = float(bin_size) / max_bin_size;
		if (retain)
		{
			parallel_for_each(acc_view, concurrency::extent<2>(src_array.get_extent()[0] / 4, src_array.get_extent()[1]).tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static int hist[bank_count][max_bin_size + 1];
				int index = idx.local[0] * tile_size + idx.local[1];
				int bank_id = index % bank_count;
				if (index <= bin_size)
				{
					for (int i = 0; i < bank_count; i += 4)
					{
						hist[i][index] = 0;
						hist[i + 1][index] = 0;
						hist[i + 2][index] = 0;
						hist[i + 3][index] = 0;
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				int src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 1, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 2, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 3, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				idx.barrier.wait_with_all_memory_fence();

				if (index < bin_size)
				{
					int total_local_hist = 0;
					for (int i = 0; i < bank_count; i += 4)
					{
						total_local_hist += hist[i][index];
						total_local_hist += hist[i + 1][index];
						total_local_hist += hist[i + 2][index];
						total_local_hist += hist[i + 3][index];
					}
					concurrency::atomic_fetch_add(&dest_array[index], total_local_hist);
				}
			});
		}
		else
		{
			dest_array.discard_data();
			parallel_for_each(acc_view, concurrency::extent<2>(src_array.get_extent()[0] / 4, src_array.get_extent()[1]).tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static int hist[bank_count][max_bin_size + 1];
				int index = idx.local[0] * tile_size + idx.local[1];
				int bank_id = index % bank_count;
				if (index <= bin_size)
				{
					if (idx.tile == concurrency::index<2>(0, 0))
					{
						dest_array[index] = 0;
					}
					for (int i = 0; i < bank_count; i += 4)
					{
						hist[i][index] = 0;
						hist[i + 1][index] = 0;
						hist[i + 2][index] = 0;
						hist[i + 3][index] = 0;
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				int src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 1, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 2, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				src_val = direct3d::clamp(int(guarded_read(src_array, concurrency::index<2>(idx.global[0] * 4 + 3, idx.global[1]), float(max_bin_size)) * bin_ratio + 0.001f), 0, bin_size);
				concurrency::atomic_fetch_inc(&hist[bank_id][src_val]);
				idx.barrier.wait_with_all_memory_fence();

				if (index < bin_size)
				{
					int total_local_hist = 0;
					for (int i = 0; i < bank_count; i += 4)
					{
						total_local_hist += hist[i][index];
						total_local_hist += hist[i + 1][index];
						total_local_hist += hist[i + 2][index];
						total_local_hist += hist[i + 3][index];
					}
					concurrency::atomic_fetch_add(&dest_array[index], total_local_hist);
				}
			});
		}
	}
}
