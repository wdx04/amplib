#pragma once

#include "amp_core.h"

namespace amp
{
	// Look-up Table
	template<typename source_t>
	inline void lut_32f_c1(accelerator_view& acc_view, array_view<source_t, 2> src_array, array_view<const float, 1> lut, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		dest_array.discard_data();
		int lut_max = lut.get_extent()[0] - 1;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int index = direct3d::clamp(int(guarded_read(src_array, idx.global)), 0, lut_max);
			guarded_write(dest_array, idx.global, lut(index));
		});
	}
}
