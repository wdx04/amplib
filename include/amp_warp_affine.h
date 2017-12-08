#pragma once

#include "amp_core.h"

namespace amp
{
	// Warp Affine
	inline void warp_affine_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, kernel_wrapper<float, 6U> M, float border_value = 0.0f, bool inverse_M = false)
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
			double D = M.data[0] * M.data[4] - M.data[1] * M.data[3];
			D = D != 0 ? 1. / D : 0;
			double A11 = M.data[4] * D, A22 = M.data[0] * D;
			M.data[0] = float(A11); M.data[1] = float(M.data[1] * -D);
			M.data[3] *= float(-D); M.data[4] = float(A22);
			double b1 = -M.data[0] * M.data[2] - M.data[1] * M.data[5];
			double b2 = -M.data[3] * M.data[2] - M.data[4] * M.data[5];
			M.data[2] = float(b1); M.data[5] = float(b2);
		}
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int dx = idx.global[1];
			int dy0 = idx.global[0];

			if(dx < dst_cols)
			{
				int tmp = dx << ab_bits;
				int X0_ = int(M.data[0] * tmp + 0.5f);
				int Y0_ = int(M.data[3] * tmp + 0.5f);

				for(int dy = dy0, dy1 = direct3d::imin(dst_rows, dy0 + 1); dy < dy1; ++dy)
				{
					int X0 = X0_ + int(direct3d::mad(M.data[1], float(dy), M.data[2]) * ab_scale + 0.5f) + round_delta;
					int Y0 = Y0_ + int(direct3d::mad(M.data[4], float(dy), M.data[5]) * ab_scale + 0.5f) + round_delta;
					X0 = X0 >> (ab_bits - inter_bits);
					Y0 = Y0 >> (ab_bits - inter_bits);

					int sx = direct3d::clamp(X0 >> inter_bits, -32768, 32767), sy = direct3d::clamp(Y0 >> inter_bits, -32768, 32767);
					int ax = X0 & (inter_tab_size - 1), ay = Y0 & (inter_tab_size - 1);

					float v0 = border_value, v1 = border_value, v2 = border_value, v3 = border_value;
					if(sx >= 0 && sx < src_cols)
					{
						if(sy >= 0 && sy < src_rows)
							v0 = src_array(sy, sx);
						if(sy + 1 >= 0 && sy + 1 < src_rows)
							v2 = src_array(sy + 1, sx);
					}
					if(sx + 1 >= 0 && sx + 1 < src_cols)
					{
						if(sy >= 0 && sy < src_rows)
							v1 = src_array(sy, sx + 1);
						if(sy + 1 >= 0 && sy + 1 < src_rows)
							v3 = src_array(sy + 1, sx + 1);
					}

					float taby = 1.0f / inter_tab_size*ay;
					float tabx = 1.0f / inter_tab_size*ax;
					float tabx2 = 1.0f - tabx, taby2 = 1.0f - taby;
					float val = direct3d::mad(tabx2, direct3d::mad(v0, taby2, v2 * taby), tabx * direct3d::mad(v1, taby2, v3 * taby));
					guarded_write(dest_array, idx.global, val);
				}
			}
		});
	}

	inline void warp_affine_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const cv::Mat& M, float border_value = 0.0f, bool inverse_M = false)
	{
		assert(M.cols == 3 && M.rows == 2);
		kernel_wrapper<float, 6U> wrapped_M(M);
		warp_affine_linear_32f_c1(acc_view, src_array, dest_array, wrapped_M, border_value, inverse_M);
	}
}
