#pragma once

#include "amp_core.h"

namespace amp
{
	// Remap
	struct coeffs_wrapper
	{
		float val[64];

		coeffs_wrapper()
		{
			std::vector<float> temp_val{ 1.000000f, 0.000000f, 0.968750f, 0.031250f, 0.937500f, 0.062500f, 0.906250f, 0.093750f, 0.875000f, 0.125000f, 0.843750f, 0.156250f,
				0.812500f, 0.187500f, 0.781250f, 0.218750f, 0.750000f, 0.250000f, 0.718750f, 0.281250f, 0.687500f, 0.312500f, 0.656250f, 0.343750f,
				0.625000f, 0.375000f, 0.593750f, 0.406250f, 0.562500f, 0.437500f, 0.531250f, 0.468750f, 0.500000f, 0.500000f, 0.468750f, 0.531250f,
				0.437500f, 0.562500f, 0.406250f, 0.593750f, 0.375000f, 0.625000f, 0.343750f, 0.656250f, 0.312500f, 0.687500f, 0.281250f, 0.718750f,
				0.250000f, 0.750000f, 0.218750f, 0.781250f, 0.187500f, 0.812500f, 0.156250f, 0.843750f, 0.125000f, 0.875000f, 0.093750f, 0.906250f,
				0.062500f, 0.937500f, 0.031250f, 0.968750f };
			memcpy(val, &temp_val[0], sizeof(float) * 64);
		}
	};

	inline void remap_linear_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<const float, 2> map1_array, array_view<const float, 2> map2_array
		, float border_value)
	{
		static const int tile_size = 32;
		static const int inter_bits = 5;
		static const int inter_tab_size = 1 << inter_bits;
		const int src_rows = src_array.get_extent()[0];
		const int src_cols = src_array.get_extent()[1];
		const int dst_rows = dest_array.get_extent()[0];
		const int dst_cols = dest_array.get_extent()[1];
		const float inter_tab_size_f = float(inter_tab_size);
		coeffs_wrapper coeffs;
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float xf = guarded_read(map1_array, idx.global), yf = guarded_read(map2_array, idx.global);
			// was convert_int_sat_rtz
			int sx = int(direct3d::mad(xf, inter_tab_size_f, 0.5f)) >> inter_bits;
			int sy = int(direct3d::mad(yf, inter_tab_size_f, 0.5f)) >> inter_bits;
			// was convert_int_rte
			int coeffs_x_index = (int(fast_math::roundf(xf * inter_tab_size_f)) & (inter_tab_size - 1)) << 1;
			int coeffs_y_index = (int(fast_math::roundf(yf * inter_tab_size_f)) & (inter_tab_size - 1)) << 1;
			float sum = (guarded_read(src_array, concurrency::index<2>(sy, sx), border_value) * coeffs.val[coeffs_x_index] + guarded_read(src_array, concurrency::index<2>(sy, sx + 1), border_value) * coeffs.val[coeffs_x_index + 1]) * coeffs.val[coeffs_y_index]
				+ (guarded_read(src_array, concurrency::index<2>(sy + 1, sx), border_value) * coeffs.val[coeffs_x_index] + guarded_read(src_array, concurrency::index<2>(sy + 1, sx + 1), border_value) * coeffs.val[coeffs_x_index + 1]) * coeffs.val[coeffs_y_index + 1];
			guarded_write(dest_array, idx.global, sum);
		});
	}
}
