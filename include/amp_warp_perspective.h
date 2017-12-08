#pragma once

#include "amp_core.h"

namespace amp
{
	// Warp Perspective
	inline void warp_perspective_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, kernel_wrapper<float, 9U> M, float border_value = 0.0f, bool inverse_M = false)
	{
		static const int tile_size = 32;
		dest_array.discard_data();
		static const int inter_bits = 5;
		static const int inter_tab_size = (1 << inter_bits);
		static const float inter_scale = 1.0f / inter_tab_size;
		static const int ab_bits = 10;
		static const int ab_scale = (1 << ab_bits);
		static const int round_delta = (1 << (ab_bits - inter_bits - 1));
		int src_rows = src_array.get_extent()[0];
		int src_cols = src_array.get_extent()[1];
		int dst_rows = dest_array.get_extent()[0];
		int dst_cols = dest_array.get_extent()[1];
		if(!inverse_M)
		{
			cv::Mat tempM(3, 3, CV_32FC1, (void *)M.data, sizeof(float) * 3);
			cv::invert(tempM, tempM);
		}
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int dx = idx.global[1];
			int dy = idx.global[0];

			if(dx < dst_cols && dy < dst_rows)
			{
				float X0 = M.data[0] * dx + M.data[1] * dy + M.data[2];
				float Y0 = M.data[3] * dx + M.data[4] * dy + M.data[5];
				float W = M.data[6] * dx + M.data[7] * dy + M.data[8];
				W = W != 0.0f ? inter_tab_size / W : 0.0f;
				int X = int(X0 * W + 0.5f), Y = int(Y0 * W + 0.5f);

				int sx = direct3d::clamp(X >> inter_bits, -32768, 32767);
				int sy = direct3d::clamp(Y >> inter_bits, -32768, 32767);
				int ay = Y & (inter_tab_size - 1);
				int ax = X & (inter_tab_size - 1);

				float v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
					src_array(sy, sx) : border_value;
				float v1 = (sx + 1 >= 0 && sx + 1 < src_cols && sy >= 0 && sy < src_rows) ?
					src_array(sy, sx + 1) : border_value;
				float v2 = (sx >= 0 && sx < src_cols && sy + 1 >= 0 && sy + 1 < src_rows) ?
					src_array(sy + 1, sx) : border_value;
				float v3 = (sx + 1 >= 0 && sx + 1 < src_cols && sy + 1 >= 0 && sy + 1 < src_rows) ?
					src_array(sy + 1, sx + 1) : border_value;

				float taby = 1.0f / inter_tab_size*ay;
				float tabx = 1.0f / inter_tab_size*ax;
				float tabx2 = 1.0f - tabx, taby2 = 1.0f - taby;
				float val = v0 * tabx2 * taby2 + v1 * tabx * taby2 + v2 * tabx2 * taby + v3 * tabx * taby;
				dest_array(idx.global) = val;
			}
		});
	}

	inline void warp_perspective_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const cv::Mat& M, float border_value = 0.0f, bool inverse_M = false)
	{
		assert(M.cols == 3 && M.rows == 3);
		kernel_wrapper<float, 9U> wrapped_M(M);
		warp_perspective_linear_32f_c1(acc_view, src_array, dest_array, wrapped_M, border_value, inverse_M);
	}
}
