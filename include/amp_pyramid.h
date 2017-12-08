#pragma once

#include "amp_core.h"

namespace amp
{
	inline int idx_row_low(int y, int last_row) restrict(amp)
	{
		return direct3d::abs(y) % (last_row + 1);
	}

	inline int idx_row_high(int y, int last_row) restrict(amp)
	{
		return direct3d::abs(last_row - (int)direct3d::abs(last_row - y)) % (last_row + 1);
	}

	inline int idx_row(int y, int last_row) restrict(amp)
	{
		return idx_row_low(idx_row_high(y, last_row), last_row);
	}

	inline int idx_col_low(int x, int last_col) restrict(amp)
	{
		return direct3d::abs(x) % (last_col + 1);
	}

	inline int idx_col_high(int x, int last_col) restrict(amp)
	{
		return direct3d::abs(last_col - (int)direct3d::abs(last_col - x)) % (last_col + 1);
	}

	inline int idx_col(int x, int last_col) restrict(amp)
	{
		return idx_col_low(idx_col_high(x, last_col), last_col);
	}

	inline void pyrdown_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 256;
		int src_rows = src_array.get_extent()[0];
		int src_cols = src_array.get_extent()[1];
		int dst_rows = dest_array.get_extent()[0];
		int dst_cols = dest_array.get_extent()[1];
		concurrency::extent<2> ext(dst_rows, src_cols);
		parallel_for_each(acc_view, ext.tile<1, tile_size>().pad(), [=](tiled_index<1, tile_size> idx) restrict(amp)
		{
			const int x = idx.global[1];
			const int y = idx.global[0];
			int local_x = idx.local[1];

			tile_static float smem[256 + 4];

			float sum;

			const int src_y = 2 * y;
			const int last_row = src_rows - 1;
			const int last_col = src_cols - 1;

			bool is_inner = src_y >= 2 && src_y < src_rows - 2 && x >= 2 && x < src_cols - 2;
			if(is_inner)
			{
				sum = 0.0625f * src_array(src_y - 2, x);
				sum = sum + 0.25f   * src_array(src_y - 1, x);
				sum = sum + 0.375f  * src_array(src_y, x);
				sum = sum + 0.25f   * src_array(src_y + 1, x);
				sum = sum + 0.0625f * src_array(src_y + 2, x);

				smem[2 + local_x] = sum;

				if(local_x < 2)
				{
					const int left_x = x - 2;

					sum = 0.0625f * src_array(src_y - 2, left_x);
					sum = sum + 0.25f   * src_array(src_y - 1, left_x);
					sum = sum + 0.375f  * src_array(src_y, left_x);
					sum = sum + 0.25f   * src_array(src_y + 1, left_x);
					sum = sum + 0.0625f * src_array(src_y + 2, left_x);

					smem[local_x] = sum;
				}

				if(local_x > 253)
				{
					const int right_x = x + 2;

					sum = 0.0625f * src_array(src_y - 2, right_x);
					sum = sum + 0.25f   * src_array(src_y - 1, right_x);
					sum = sum + 0.375f  * src_array(src_y, right_x);
					sum = sum + 0.25f   * src_array(src_y + 1, right_x);
					sum = sum + 0.0625f * src_array(src_y + 2, right_x);

					smem[4 + local_x] = sum;
				}
			}
			else
			{
				int col = idx_col(x, last_col);

				sum = 0.0625f * src_array(idx_row(src_y - 2, last_row), col);
				sum = sum + 0.25f   * src_array(idx_row(src_y - 1, last_row), col);
				sum = sum + 0.375f  * src_array(idx_row(src_y, last_row), col);
				sum = sum + 0.25f   *src_array(idx_row(src_y + 1, last_row), col);
				sum = sum + 0.0625f * src_array(idx_row(src_y + 2, last_row), col);

				smem[2 + local_x] = sum;

				if(local_x < 2)
				{
					const int left_x = x - 2;

					col = idx_col(left_x, last_col);

					sum = 0.0625f * src_array(idx_row(src_y - 2, last_row), col);
					sum = sum + 0.25f   * src_array(idx_row(src_y - 1, last_row), col);
					sum = sum + 0.375f  * src_array(idx_row(src_y, last_row), col);
					sum = sum + 0.25f   * src_array(idx_row(src_y + 1, last_row), col);
					sum = sum + 0.0625f * src_array(idx_row(src_y + 2, last_row), col);

					smem[local_x] = sum;
				}

				if(local_x > 253)
				{
					const int right_x = x + 2;

					col = idx_col(right_x, last_col);

					sum = 0.0625f * src_array(idx_row(src_y - 2, last_row), col);
					sum = sum + 0.25f   * src_array(idx_row(src_y - 1, last_row), col);
					sum = sum + 0.375f  * src_array(idx_row(src_y, last_row), col);
					sum = sum + 0.25f   * src_array(idx_row(src_y + 1, last_row), col);
					sum = sum + 0.0625f * src_array(idx_row(src_y + 2, last_row), col);

					smem[4 + local_x] = sum;
				}
			}

			idx.barrier.wait_with_tile_static_memory_fence();

			if(local_x < 128)
			{
				const int tid2 = local_x * 2;

				sum = 0.0625f * smem[2 + tid2 - 2];
				sum = sum + 0.25f   * smem[2 + tid2 - 1];
				sum = sum + 0.375f  * smem[2 + tid2];
				sum = sum + 0.25f   * smem[2 + tid2 + 1];
				sum = sum + 0.0625f * smem[2 + tid2 + 2];

				const int dst_x = (idx.tile[1] * idx.tile_dim1 + tid2) / 2;

				if(dst_x < dst_cols)
				{
					dest_array(y, dst_x) = sum;
				}
			}
		});
	}

	inline void pyrup_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 16;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			const int x = idx.global[1];
			const int y = idx.global[0];

			const int lsizex = idx.tile_dim1;
			const int lsizey = idx.tile_dim0;

			const int tidx = idx.local[1];
			const int tidy = idx.local[0];

			tile_static float s_srcPatch[10][10];
			tile_static float s_dstPatch[20][16];

			if(tidx < 10 && tidy < 10)
			{
				int srcx = direct3d::mad((int)idx.tile[1], lsizex >> 1, tidx) - 1;
				int srcy = direct3d::mad((int)idx.tile[0], lsizey >> 1, tidy) - 1;

				srcx = direct3d::abs(srcx);
				srcx = direct3d::imin(src_array.get_extent()[1] - 1, srcx);

				srcy = direct3d::abs(srcy);
				srcy = direct3d::imin(src_array.get_extent()[0] - 1, srcy);

				s_srcPatch[tidy][tidx] = src_array(srcy, srcx);
			}

			idx.barrier.wait_with_tile_static_memory_fence();

			float sum = 0.0f;
			const float evenFlag = (float)((tidx & 1) == 0);
			const float oddFlag = (float)((tidx & 1) != 0);
			const bool  eveny = ((tidy & 1) == 0);

			const float co1 = (float)0.375f;
			const float co2 = (float)0.25f;
			const float co3 = (float)0.0625f;

			if(eveny)
			{
				sum = (evenFlag* co3) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx - 2) >> 1)];
				sum = sum + (oddFlag * co2) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx - 1) >> 1)];
				sum = sum + (evenFlag* co1) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx) >> 1)];
				sum = sum + (oddFlag * co2) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx + 1) >> 1)];
				sum = sum + (evenFlag* co3) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx + 2) >> 1)];
			}

			s_dstPatch[2 + tidy][tidx] = sum;

			if(tidy < 2)
			{
				sum = 0;

				if(eveny)
				{
					sum = (evenFlag * co3) * s_srcPatch[lsizey - 16][1 + ((tidx - 2) >> 1)];
					sum = sum + (oddFlag * co2) * s_srcPatch[lsizey - 16][1 + ((tidx - 1) >> 1)];
					sum = sum + (evenFlag * co1) * s_srcPatch[lsizey - 16][1 + ((tidx) >> 1)];
					sum = sum + (oddFlag * co2) * s_srcPatch[lsizey - 16][1 + ((tidx + 1) >> 1)];
					sum = sum + (evenFlag * co3) * s_srcPatch[lsizey - 16][1 + ((tidx + 2) >> 1)];
				}

				s_dstPatch[tidy][tidx] = sum;
			}

			if(tidy > 13)
			{
				sum = 0;

				if(eveny)
				{
					sum = (evenFlag * co3) * s_srcPatch[lsizey - 7][1 + ((tidx - 2) >> 1)];
					sum = sum + (oddFlag * co2) * s_srcPatch[lsizey - 7][1 + ((tidx - 1) >> 1)];
					sum = sum + (evenFlag * co1) * s_srcPatch[lsizey - 7][1 + ((tidx) >> 1)];
					sum = sum + (oddFlag * co2) * s_srcPatch[lsizey - 7][1 + ((tidx + 1) >> 1)];
					sum = sum + (evenFlag * co3) * s_srcPatch[lsizey - 7][1 + ((tidx + 2) >> 1)];
				}
				s_dstPatch[4 + tidy][tidx] = sum;
			}

			idx.barrier.wait_with_tile_static_memory_fence();

			sum = co3 * s_dstPatch[2 + tidy - 2][tidx];
			sum = sum + co2 * s_dstPatch[2 + tidy - 1][tidx];
			sum = sum + co1 * s_dstPatch[2 + tidy][tidx];
			sum = sum + co2 * s_dstPatch[2 + tidy + 1][tidx];
			sum = sum + co3 * s_dstPatch[2 + tidy + 2][tidx];

			guarded_write(dest_array, concurrency::index<2>(y, x), 4.0f * sum);
		});
	}
}
