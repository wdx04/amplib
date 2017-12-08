#pragma once

#include "amp_core.h"

namespace amp
{
	// Contrast-limited Adaptive Histogram Equalization
	template<int tile_size_y, int tile_size_x>
	static inline int clahe_calc_lut(tiled_index<tile_size_y, tile_size_x> idx, int *smem, int val, int tid) restrict(amp)
	{
		smem[tid] = val;
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid == 0)
		{
			for(int i = 1; i < 256; ++i)
			{
				smem[i] += smem[i - 1];
			}
		}
		idx.barrier.wait_with_tile_static_memory_fence();
		return smem[tid];
	}

	template<int tile_size_y, int tile_size_x>
	static inline void clahe_reduce(tiled_index<tile_size_y, tile_size_x> idx, int *smem, int val, int tid) restrict(amp)
	{
		smem[tid] = val;
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid < 128)
			smem[tid] = val += smem[tid + 128];
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid < 64)
			smem[tid] = val += smem[tid + 64];
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid < 32)
			smem[tid] += smem[tid + 32];
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid < 8)
		{
			smem[tid] += (smem[tid + 8] + smem[tid + 16] + smem[tid + 24]);
		}
		idx.barrier.wait_with_tile_static_memory_fence();
		if(tid == 0)
		{
			smem[0] += (smem[1] + smem[2] + smem[3] + smem[4] + smem[5] + smem[6] + smem[7]);
		}
	}

	inline void clahe_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, float clip_limit = 40.0f, int grid_count_x = 8, int grid_count_y = 8)
	{
		const int hist_size = 256;
		const int dest_rows = dest_array.get_extent()[0];
		const int dest_cols = dest_array.get_extent()[1];
		const int grid_size_x = DIVUP(dest_cols, grid_count_x);
		const int grid_size_y = DIVUP(dest_rows, grid_count_y);
		const int grid_size_total = grid_size_x * grid_size_y;
		const float lutScale = static_cast<float>(hist_size - 1) / grid_size_total;
		int clip_limit_i = 0;
		if(clip_limit > 0.0f)
		{
			clip_limit_i = static_cast<int>(clip_limit * grid_size_total / hist_size);
			clip_limit_i = std::max<int>(clip_limit_i, 1);
		}
		static const int tile_size_x = 32;
		static const int tile_size_y = 8;
		concurrency::extent<2> ext(tile_size_y * grid_count_y, tile_size_x * grid_count_x);
		concurrency::array<int, 3> lut(grid_count_y, grid_count_x, hist_size, acc_view);
		array_view<int, 3> lut_view(lut);
		lut_view.discard_data();
		parallel_for_each(acc_view, ext.tile<tile_size_y, tile_size_x>().pad(), [=](tiled_index<tile_size_y, tile_size_x> idx) restrict(amp)
		{
			tile_static int smem[512];

			int tx = idx.tile[1];
			int ty = idx.tile[0];
			int tid = idx.local[0] * idx.tile_dim1 + idx.local[1];
			smem[tid] = 0;
			idx.barrier.wait_with_tile_static_memory_fence();

			for(int i = idx.local[0]; i < grid_size_y; i += idx.tile_dim0)
			{
				for(int j = idx.local[1]; j < grid_size_x; j += idx.tile_dim1)
				{
					const int data = direct3d::clamp(int(guarded_read_reflect101(src_array, concurrency::index<2>(ty * grid_size_y + i, tx * grid_size_x + j)) + 0.001f), 0, hist_size - 1);
					concurrency::atomic_fetch_inc(&smem[data]);
				}
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			int tHistVal = smem[tid];
			idx.barrier.wait_with_tile_static_memory_fence();

			if(clip_limit_i > 0)
			{
				// clip histogram bar
				int clipped = 0;
				if(tHistVal > clip_limit_i)
				{
					clipped = tHistVal - clip_limit_i;
					tHistVal = clip_limit_i;
				}

				// find number of overall clipped samples
				clahe_reduce(idx, smem, clipped, tid);
				idx.barrier.wait_with_tile_static_memory_fence();
				clipped = smem[0];

				// broadcast evaluated value
				tile_static int totalClipped;
				if(tid == 0)
				{
					totalClipped = clipped;
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				// redistribute clipped samples evenly
				int redistBatch = totalClipped / hist_size;
				tHistVal += redistBatch;
				int residual = totalClipped - redistBatch * hist_size;
				if(tid < residual)
					++tHistVal;
			}

			const int lutVal = clahe_calc_lut(idx, smem, tHistVal, tid);
			int ires = int(fast_math::roundf(lutScale * lutVal));
			guarded_write(lut_view, concurrency::index<3>(ty, tx, tid), direct3d::clamp(ires, 0, hist_size - 1));
		});
		dest_array.discard_data();
		array_view<const int, 3> lut_const_view(lut);
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size_y, tile_size_x>().pad(), [=](tiled_index<tile_size_y, tile_size_x> idx) restrict(amp)
		{
			const int x = idx.global[1];
			const int y = idx.global[0];

			const float tyf = (float(y) / grid_size_y) - 0.5f;
			int ty1 = int(fast_math::floorf(tyf)); // was rtn
			int ty2 = ty1 + 1;
			const float ya = tyf - ty1;
			ty1 = direct3d::imax(ty1, 0);
			ty2 = direct3d::imin(ty2, grid_count_y - 1);

			const float txf = (float(x) / grid_size_x) - 0.5f;
			int tx1 = int(fast_math::floorf(txf)); // was rtn
			int tx2 = tx1 + 1;
			const float xa = txf - tx1;
			tx1 = direct3d::imax(tx1, 0);
			tx2 = direct3d::imin(tx2, grid_count_x - 1);

			const int srcVal = direct3d::clamp(int(fast_math::roundf(guarded_read(src_array, idx.global))), 0, hist_size - 1);
			float res = 0.0f;
			res += lut_const_view(ty1, tx1, srcVal) * ((1.0f - xa) * (1.0f - ya));
			res += lut_const_view(ty1, tx2, srcVal) * ((xa)* (1.0f - ya));
			res += lut_const_view(ty2, tx1, srcVal) * ((1.0f - xa) * (ya));
			res += lut_const_view(ty2, tx2, srcVal) * ((xa)* (ya));
			guarded_write(dest_array, idx.global, res);
		});
	}
	
}
