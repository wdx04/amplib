#pragma once

#include "amp_core.h"
#include "amp_geodesic_dilate.h"

namespace amp
{
	// Delete Border Components
	inline void delete_border_components_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array)
	{
		// 1.create edge image
		static const int tile_size = 32;
		parallel_for_each(acc_view, temp_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			bool on_edge = gidx[0] == 0 || gidx[1] == 0 || gidx[0] == temp_array.get_extent()[0] - 1 || gidx[1] == temp_array.get_extent()[1] - 1;
			guarded_write(temp_array, gidx, (on_edge ? guarded_read(src_array, gidx) : 0.0f));
		});
		// 2.reconstruct
		geodesic_dilate_32f_c1(acc_view, temp_array, dest_array, src_array, temp_array);
		// 3.subtract
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_array.get_extent().contains(idx.global))
			{
				dest_array(idx.global) = src_array(idx.global) - dest_array(idx.global);
			}
		});
	}

}