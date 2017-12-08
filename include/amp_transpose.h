#pragma once

#include "amp_core.h"

namespace amp
{
	// Transpose
	inline void transpose_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		assert(src_array.get_extent()[1] == dest_array.get_extent()[0]);
		assert(src_array.get_extent()[0] == dest_array.get_extent()[1]);
		dest_array.discard_data();
		static const int tile_size = 32;
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float local_data[tile_size][tile_size + 1];
			local_data[idx.local[1]][idx.local[0]] = guarded_read(src_array, idx.global);
			idx.barrier.wait();

			guarded_write(dest_array, concurrency::index<2>(idx.tile_origin[1], idx.tile_origin[0]) + idx.local, local_data[idx.local[0]][idx.local[1]]);
		});
	}
}
