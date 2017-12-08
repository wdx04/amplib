#pragma once

#include "amp_core.h"

namespace amp
{
	inline void calc_ltp_r1_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array_1, array_view<float, 2> dest_array_2, float thresh)
	{
		static const int tile_size = 32;
		static const int tile_static_size_3x3 = tile_size + 2;
		dest_array_1.discard_data();
		dest_array_2.discard_data();
		parallel_for_each(acc_view, dest_array_1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
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
			data[dr][dc] = amp::guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_3x3 / 2][dc] = amp::guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result_1 = 0;
			unsigned int result_2 = 0;
			float center_val = data[local_row + 1][local_col + 1];
			result_1 += (center_val + thresh <= data[local_row + 0][local_col + 0] ? 1 : 0);
			result_1 += (center_val + thresh <= data[local_row + 0][local_col + 1] ? 2 : 0);
			result_1 += (center_val + thresh <= data[local_row + 0][local_col + 2] ? 4 : 0);
			result_1 += (center_val + thresh <= data[local_row + 1][local_col + 2] ? 8 : 0);
			result_1 += (center_val + thresh <= data[local_row + 2][local_col + 2] ? 16 : 0);
			result_1 += (center_val + thresh <= data[local_row + 2][local_col + 1] ? 32 : 0);
			result_1 += (center_val + thresh <= data[local_row + 2][local_col + 0] ? 64 : 0);
			result_1 += (center_val + thresh <= data[local_row + 1][local_col + 0] ? 128 : 0);
			result_2 += (center_val >= data[local_row + 0][local_col + 0] + thresh ? 1 : 0);
			result_2 += (center_val >= data[local_row + 0][local_col + 1] + thresh ? 2 : 0);
			result_2 += (center_val >= data[local_row + 0][local_col + 2] + thresh ? 4 : 0);
			result_2 += (center_val >= data[local_row + 1][local_col + 2] + thresh ? 8 : 0);
			result_2 += (center_val >= data[local_row + 2][local_col + 2] + thresh ? 16 : 0);
			result_2 += (center_val >= data[local_row + 2][local_col + 1] + thresh ? 32 : 0);
			result_2 += (center_val >= data[local_row + 2][local_col + 0] + thresh ? 64 : 0);
			result_2 += (center_val >= data[local_row + 1][local_col + 0] + thresh ? 128 : 0);

			amp::guarded_write(dest_array_1, idx.global, float(result_1));
			amp::guarded_write(dest_array_2, idx.global, float(result_2));
		});
	}

	inline void calc_ltp_r3_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array_1, array_view<float, 2> dest_array_2, float thresh)
	{
		static const int tile_size = 32;
		static const int tile_static_size_7x7 = tile_size + 6;
		dest_array_1.discard_data();
		dest_array_2.discard_data();
		parallel_for_each(acc_view, dest_array_1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float data[tile_static_size_7x7][tile_static_size_7x7];
			int row = idx.global[0];
			int col = idx.global[1];
			int local_row = idx.local[0];
			int local_col = idx.local[1];
			int dx = row - local_row - 3;
			int dy = col - local_col - 3;
			const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_7x7 * tile_static_size_7x7 / 2 - 1);
			int dr = id / tile_static_size_7x7;
			int dc = id % tile_static_size_7x7;
			data[dr][dc] = amp::guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_7x7 / 2][dc] = amp::guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_7x7 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result_1 = 0;
			unsigned int result_2 = 0;
			float center_val = data[local_row + 3][local_col + 3];
			result_1 += (center_val + thresh <= data[local_row + 0][local_col + 3] ? 1 : 0);
			result_1 += (center_val + thresh <= data[local_row + 1][local_col + 1] ? 2 : 0);
			result_1 += (center_val + thresh <= data[local_row + 3][local_col + 0] ? 4 : 0);
			result_1 += (center_val + thresh <= data[local_row + 5][local_col + 1] ? 8 : 0);
			result_1 += (center_val + thresh <= data[local_row + 6][local_col + 3] ? 16 : 0);
			result_1 += (center_val + thresh <= data[local_row + 5][local_col + 5] ? 32 : 0);
			result_1 += (center_val + thresh <= data[local_row + 3][local_col + 6] ? 64 : 0);
			result_1 += (center_val + thresh <= data[local_row + 1][local_col + 5] ? 128 : 0);
			result_2 += (center_val >= data[local_row + 0][local_col + 3] + thresh ? 1 : 0);
			result_2 += (center_val >= data[local_row + 1][local_col + 1] + thresh ? 2 : 0);
			result_2 += (center_val >= data[local_row + 3][local_col + 0] + thresh ? 4 : 0);
			result_2 += (center_val >= data[local_row + 5][local_col + 1] + thresh ? 8 : 0);
			result_2 += (center_val >= data[local_row + 6][local_col + 3] + thresh ? 16 : 0);
			result_2 += (center_val >= data[local_row + 5][local_col + 5] + thresh ? 32 : 0);
			result_2 += (center_val >= data[local_row + 3][local_col + 6] + thresh ? 64 : 0);
			result_2 += (center_val >= data[local_row + 1][local_col + 5] + thresh ? 128 : 0);

			amp::guarded_write(dest_array_1, idx.global, float(result_1));
			amp::guarded_write(dest_array_2, idx.global, float(result_2));
		});
	}

}
