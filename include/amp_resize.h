#pragma once

#include "amp_core.h"

namespace amp
{
	// Resize
	inline void resize_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		float fx = float(dest_array.get_extent()[1]) / src_array.get_extent()[1];
		float fy = float(dest_array.get_extent()[0]) / src_array.get_extent()[0];
		float ifx = 1.0f / fx, ify = 1.0f / fy;
		int src_cols = src_array.get_extent()[1];
		int src_rows = src_array.get_extent()[0];
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int dx = idx.global[1];
			int dy = idx.global[0];

			float sx = ((dx + 0.5f) * ifx - 0.5f), sy = ((dy + 0.5f) * ify - 0.5f);
			int x = int(sx), y = int(sy);
			float u = sx - x, v = sy - y;

			if(x<0) x = 0, u = 0.0f;
			if(x >= src_cols) x = src_cols - 1, u = 0.0f;
			if(y<0) y = 0, v = 0.0f;
			if(y >= src_rows) y = src_rows - 1, v = 0.0f;
			int y_ = direct3d::imin(y + 1, src_rows - 1);
			int x_ = direct3d::imin(x + 1, src_cols - 1);
			float u1 = 1.0f - u;
			float v1 = 1.0f - v;
			float data0 = src_array(y, x);
			float data1 = src_array(y, x_);
			float data2 = src_array(y_, x);
			float data3 = src_array(y_, x_);
			guarded_write(dest_array, idx.global, u1 * v1 * data0 + u * v1 * data1 + u1 * v *data2 + u * v *data3);
		});
	}
}
