#pragma once

#include "amp_core.h"

namespace amp
{
	// Make Border
	enum class border_type { reflect_101 = 0, replicate = 1, constant = 2 };
	inline void copy_make_border_32f_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray, array_view<float, 2> destArray, border_type type = border_type::constant, float border_value = 0.0f, int expand_top = -1, int expand_left = -1)
	{
		static const int tile_size = 32;
		int nExpandRows = expand_top < 0 ? (destArray.get_extent()[0] - srcArray.get_extent()[0]) / 2 : expand_top;
		int nExpandCols = expand_left < 0 ? (destArray.get_extent()[1] - srcArray.get_extent()[1]) / 2 : expand_left;
		if(type == border_type::reflect_101)
		{
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(destArray, idx.global, guarded_read_reflect101(srcArray, concurrency::index<2>(idx.global[0] - nExpandRows, idx.global[1] - nExpandCols)));
			});
		}
		else if(type == border_type::replicate)
		{
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(destArray, idx.global, guarded_read_replicate(srcArray, concurrency::index<2>(idx.global[0] - nExpandRows, idx.global[1] - nExpandCols)));
			});
		}
		else if(type == border_type::constant)
		{
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(destArray, idx.global, guarded_read(srcArray, concurrency::index<2>(idx.global[0] - nExpandRows, idx.global[1] - nExpandCols), border_value));
			});
		}
	}
}
