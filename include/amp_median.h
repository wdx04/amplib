#pragma once

#include "amp_core.h"

namespace amp
{
	// 3x3 Median Filter
#define median_op(a,b) {mid=a; a=fminf(a,b); b=fmaxf(mid,b);}
	inline void median_filter_3x3_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		using fast_math::fmaxf;
		using fast_math::fminf;
		static const int tile_size = 8;
		static const int tile_static_size = tile_size + 2;
		dest_array.discard_data();
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float data[tile_static_size][tile_static_size + 1];
			const int local_row = idx.local[0];
			const int local_col = idx.local[1];
			const int dx = idx.global[0] - local_row - 1;
			const int dy = idx.global[1] - local_col - 1;
			const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size * tile_static_size / 2 - 1);
			const int dr = id / tile_static_size;
			const int dc = id % tile_static_size;
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			float p0 = data[local_row][local_col], p1 = data[local_row][local_col + 1], p2 = data[local_row][local_col + 2];
			float p3 = data[local_row + 1][local_col], p4 = data[local_row + 1][local_col + 1], p5 = data[local_row + 1][local_col + 2];
			float p6 = data[local_row + 2][local_col], p7 = data[local_row + 2][local_col + 1], p8 = data[local_row + 2][local_col + 2];

			float mid;
			median_op(p1, p2); median_op(p4, p5); median_op(p7, p8); median_op(p0, p1);
			median_op(p3, p4); median_op(p6, p7); median_op(p1, p2); median_op(p4, p5);
			median_op(p7, p8); median_op(p0, p3); median_op(p5, p8); median_op(p4, p7);
			median_op(p3, p6); median_op(p1, p4); median_op(p2, p5); median_op(p4, p7);
			median_op(p4, p2); median_op(p6, p4); median_op(p4, p2);

			guarded_write(dest_array, idx.global, p4);
		});
	}
#undef median_op

	// 5x5 Median Filter
#define median_op(a,b) {mid=a; a=fminf(a,b); b=fmaxf(mid,b);}
	inline void median_filter_5x5_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		using fast_math::fmaxf;
		using fast_math::fminf;
		static const int tile_size = 8;
		static const int tile_static_size = tile_size + 4;
		dest_array.discard_data();
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float data[tile_static_size][tile_static_size + 1];
			const int local_row = idx.local[0];
			const int local_col = idx.local[1];
			const int dx = idx.global[0] - local_row - 2;
			const int dy = idx.global[1] - local_col - 2;
			const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size * tile_static_size / 2 - 1);
			const int dr = id / tile_static_size;
			const int dc = id % tile_static_size;
			data[dr][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr, dy + dc));
			data[dr + tile_static_size / 2][dc] = guarded_read_reflect101(src_array, concurrency::index<2>(dx + dr + tile_static_size / 2, dy + dc));
			idx.barrier.wait_with_tile_static_memory_fence();

			float p0 = data[local_row][local_col], p1 = data[local_row][local_col + 1], p2 = data[local_row][local_col + 2], p3 = data[local_row][local_col + 3], p4 = data[local_row][local_col + 4];
			float p5 = data[local_row + 1][local_col], p6 = data[local_row + 1][local_col + 1], p7 = data[local_row + 1][local_col + 2], p8 = data[local_row + 1][local_col + 3], p9 = data[local_row + 1][local_col + 4];
			float p10 = data[local_row + 2][local_col], p11 = data[local_row + 2][local_col + 1], p12 = data[local_row + 2][local_col + 2], p13 = data[local_row + 2][local_col + 3], p14 = data[local_row + 2][local_col + 4];
			float p15 = data[local_row + 3][local_col], p16 = data[local_row + 3][local_col + 1], p17 = data[local_row + 3][local_col + 2], p18 = data[local_row + 3][local_col + 3], p19 = data[local_row + 3][local_col + 4];
			float p20 = data[local_row + 4][local_col], p21 = data[local_row + 4][local_col + 1], p22 = data[local_row + 4][local_col + 2], p23 = data[local_row + 4][local_col + 3], p24 = data[local_row + 4][local_col + 4];

			float mid;
			median_op(p1, p2); median_op(p0, p1); median_op(p1, p2); median_op(p4, p5); median_op(p3, p4);
			median_op(p4, p5); median_op(p0, p3); median_op(p2, p5); median_op(p2, p3); median_op(p1, p4);
			median_op(p1, p2); median_op(p3, p4); median_op(p7, p8); median_op(p6, p7); median_op(p7, p8);
			median_op(p10, p11); median_op(p9, p10); median_op(p10, p11); median_op(p6, p9); median_op(p8, p11);
			median_op(p8, p9); median_op(p7, p10); median_op(p7, p8); median_op(p9, p10); median_op(p0, p6);
			median_op(p4, p10); median_op(p4, p6); median_op(p2, p8); median_op(p2, p4); median_op(p6, p8);
			median_op(p1, p7); median_op(p5, p11); median_op(p5, p7); median_op(p3, p9); median_op(p3, p5);
			median_op(p7, p9); median_op(p1, p2); median_op(p3, p4); median_op(p5, p6); median_op(p7, p8);
			median_op(p9, p10); median_op(p13, p14); median_op(p12, p13); median_op(p13, p14); median_op(p16, p17);
			median_op(p15, p16); median_op(p16, p17); median_op(p12, p15); median_op(p14, p17); median_op(p14, p15);
			median_op(p13, p16); median_op(p13, p14); median_op(p15, p16); median_op(p19, p20); median_op(p18, p19);
			median_op(p19, p20); median_op(p21, p22); median_op(p23, p24); median_op(p21, p23); median_op(p22, p24);
			median_op(p22, p23); median_op(p18, p21); median_op(p20, p23); median_op(p20, p21); median_op(p19, p22);
			median_op(p22, p24); median_op(p19, p20); median_op(p21, p22); median_op(p23, p24); median_op(p12, p18);
			median_op(p16, p22); median_op(p16, p18); median_op(p14, p20); median_op(p20, p24); median_op(p14, p16);
			median_op(p18, p20); median_op(p22, p24); median_op(p13, p19); median_op(p17, p23); median_op(p17, p19);
			median_op(p15, p21); median_op(p15, p17); median_op(p19, p21); median_op(p13, p14); median_op(p15, p16);
			median_op(p17, p18); median_op(p19, p20); median_op(p21, p22); median_op(p23, p24); median_op(p0, p12);
			median_op(p8, p20); median_op(p8, p12); median_op(p4, p16); median_op(p16, p24); median_op(p12, p16);
			median_op(p2, p14); median_op(p10, p22); median_op(p10, p14); median_op(p6, p18); median_op(p6, p10);
			median_op(p10, p12); median_op(p1, p13); median_op(p9, p21); median_op(p9, p13); median_op(p5, p17);
			median_op(p13, p17); median_op(p3, p15); median_op(p11, p23); median_op(p11, p15); median_op(p7, p19);
			median_op(p7, p11); median_op(p11, p13); median_op(p11, p12);

			guarded_write(dest_array, idx.global, p12);
		});
	}
#undef median_op

}
