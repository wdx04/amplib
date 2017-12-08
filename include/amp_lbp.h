#pragma once

#include "amp_core.h"

namespace amp
{
	inline void calc_lbp_r1_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		static const int tile_static_size_3x3 = tile_size + 2;
		dest_array.discard_data();
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
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_3x3 / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result = 0;
			float center_val = data[local_row + 1][local_col + 1];
			result += (center_val <= data[local_row + 0][local_col + 0] ? 1 : 0);
			result += (center_val <= data[local_row + 0][local_col + 1] ? 2 : 0);
			result += (center_val <= data[local_row + 0][local_col + 2] ? 4 : 0);
			result += (center_val <= data[local_row + 1][local_col + 2] ? 8 : 0);
			result += (center_val <= data[local_row + 2][local_col + 2] ? 16 : 0);
			result += (center_val <= data[local_row + 2][local_col + 1] ? 32 : 0);
			result += (center_val <= data[local_row + 2][local_col + 0] ? 64 : 0);
			result += (center_val <= data[local_row + 1][local_col + 0] ? 128 : 0);

			amp::guarded_write(dest_array, idx.global, float(result));
		});
	}

	inline void calc_lbp_r3_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		static const int tile_static_size_7x7 = tile_size + 6;
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
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
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_7x7 / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_7x7 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result = 0;
			float center_val = data[local_row + 3][local_col + 3];
			result += (center_val <= data[local_row + 0][local_col + 3] ? 1 : 0);
			result += (center_val <= data[local_row + 1][local_col + 1] ? 2 : 0);
			result += (center_val <= data[local_row + 3][local_col + 0] ? 4 : 0);
			result += (center_val <= data[local_row + 5][local_col + 1] ? 8 : 0);
			result += (center_val <= data[local_row + 6][local_col + 3] ? 16 : 0);
			result += (center_val <= data[local_row + 5][local_col + 5] ? 32 : 0);
			result += (center_val <= data[local_row + 3][local_col + 6] ? 64 : 0);
			result += (center_val <= data[local_row + 1][local_col + 5] ? 128 : 0);

			amp::guarded_write(dest_array, idx.global, float(result));
		});
	}

	inline void calc_rlbp_r1_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		static const int tile_static_size_3x3 = tile_size + 2;
		dest_array.discard_data();
		kernel_wrapper<int, 15U> k(1, 15, { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64 });
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
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_3x3 / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result = 0;
			unsigned int d_index = 0;
			float neighbour_vals[9];
			neighbour_vals[0] = data[local_row + 0][local_col + 1];
			neighbour_vals[1] = data[local_row + 0][local_col + 0];
			neighbour_vals[2] = data[local_row + 1][local_col + 0];
			neighbour_vals[3] = data[local_row + 2][local_col + 0];
			neighbour_vals[4] = data[local_row + 2][local_col + 1];
			neighbour_vals[5] = data[local_row + 2][local_col + 2];
			neighbour_vals[6] = data[local_row + 1][local_col + 2];
			neighbour_vals[7] = data[local_row + 0][local_col + 2];
			float center_val = data[local_row + 1][local_col + 1];
			float max_diff = 0.0f;
			for(unsigned int i = 0; i < 8; i++)
			{
				float diff = fast_math::fabsf(center_val - neighbour_vals[i]);
				if(diff > max_diff)
				{
					d_index = (8u - i) % 8u;
					max_diff = diff;
				}
			}
			result += (center_val <= neighbour_vals[0] ? k.data[d_index + 0u] : 0);
			result += (center_val <= neighbour_vals[1] ? k.data[d_index + 1u] : 0);
			result += (center_val <= neighbour_vals[2] ? k.data[d_index + 2u] : 0);
			result += (center_val <= neighbour_vals[3] ? k.data[d_index + 3u] : 0);
			result += (center_val <= neighbour_vals[4] ? k.data[d_index + 4u] : 0);
			result += (center_val <= neighbour_vals[5] ? k.data[d_index + 5u] : 0);
			result += (center_val <= neighbour_vals[6] ? k.data[d_index + 6u] : 0);
			result += (center_val <= neighbour_vals[7] ? k.data[d_index + 7u] : 0);

			guarded_write(dest_array, idx.global, float(result));
		});
	}

	inline void calc_rlbp_r3_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		static const int tile_static_size_7x7 = tile_size + 6;
		dest_array.discard_data();
		kernel_wrapper<int, 15U> k(1, 15, { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64 });
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
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
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size_7x7 / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size_7x7 / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			unsigned int result = 0;
			unsigned int d_index = 0;
			float neighbour_vals[9];
			neighbour_vals[0] = data[local_row + 0][local_col + 3];
			neighbour_vals[1] = data[local_row + 1][local_col + 1];
			neighbour_vals[2] = data[local_row + 3][local_col + 0];
			neighbour_vals[3] = data[local_row + 5][local_col + 1];
			neighbour_vals[4] = data[local_row + 6][local_col + 3];
			neighbour_vals[5] = data[local_row + 5][local_col + 5];
			neighbour_vals[6] = data[local_row + 3][local_col + 6];
			neighbour_vals[7] = data[local_row + 1][local_col + 5];
			float center_val = data[local_row + 3][local_col + 3];
			float max_diff = 0.0f;
			for(unsigned int i = 0; i < 8; i++)
			{
				float diff = fast_math::fabsf(center_val - neighbour_vals[i]);
				if(diff > max_diff)
				{
					d_index = (8u - i) % 8u;
					max_diff = diff;
				}
			}
			result += (center_val <= neighbour_vals[0] ? k.data[d_index + 0u] : 0);
			result += (center_val <= neighbour_vals[1] ? k.data[d_index + 1u] : 0);
			result += (center_val <= neighbour_vals[2] ? k.data[d_index + 2u] : 0);
			result += (center_val <= neighbour_vals[3] ? k.data[d_index + 3u] : 0);
			result += (center_val <= neighbour_vals[4] ? k.data[d_index + 4u] : 0);
			result += (center_val <= neighbour_vals[5] ? k.data[d_index + 5u] : 0);
			result += (center_val <= neighbour_vals[6] ? k.data[d_index + 6u] : 0);
			result += (center_val <= neighbour_vals[7] ? k.data[d_index + 7u] : 0);

			guarded_write(dest_array, idx.global, float(result));
		});
	}

	inline void uniform_lbp_histogram(const std::vector<int>& hist, std::vector<float>& uniform_hist, float ratio)
	{
		static const char uniform_lbp_lut[256] = {
			0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
			11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
			16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
			17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
			22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
			58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
			23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
			24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
			29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
			58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
			58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
			58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
			36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
			58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
			42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
			47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

		uniform_hist.resize(59);
		for(int i = 0; i < std::min(256, int(hist.size())); i++)
		{
			uniform_hist[uniform_lbp_lut[i]] += float(hist[i]) * ratio;
		}
	}
}
