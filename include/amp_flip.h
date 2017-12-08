#pragma once

#include "amp_core.h"

namespace amp
{
	// Flip
	enum class flip_op { row = 0, col = 1, both = 2 };
	inline void flip_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, flip_op op)
	{
		assert(src_array.get_extent()[0] == dest_array.get_extent()[0]);
		assert(src_array.get_extent()[1] == dest_array.get_extent()[1]);
		dest_array.discard_data();
		static const int tile_size = 32;
		switch(op)
		{
		case flip_op::col:
		{
			int cols = dest_array.get_extent()[1];
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, concurrency::index<2>(idx.global[0], cols - 1 - idx.global[1]));
				guarded_write(dest_array, idx.global, src_value);
			});
			break;
		}
		case flip_op::row:
		{
			int rows = dest_array.get_extent()[0];
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, concurrency::index<2>(rows - 1 - idx.global[0], idx.global[1]));
				guarded_write(dest_array, idx.global, src_value);
			});
			break;
		}
		case flip_op::both:
		{
			int rows = dest_array.get_extent()[0];
			int cols = dest_array.get_extent()[1];
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, concurrency::index<2>(rows - 1 - idx.global[0], cols - 1 - idx.global[1]));
				guarded_write(dest_array, idx.global, src_value);
			});
			break;
		}
		}
	}
}
