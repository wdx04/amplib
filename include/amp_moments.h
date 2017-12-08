#pragma once

#include "amp_core.h"

namespace amp
{
	struct moments
	{
		unsigned int m00, m01, m10, m02, m11, m20;

		void clear() restrict(amp, cpu)
		{
			m00 = 0; m01 = 0; m10 = 0; m02 = 0; m11 = 0; m20 = 0;
		}
	};

	inline void calc_moments(accelerator_view& acc_view, array_view<const int, 2> label_array, concurrency::array_view<moments, 1> moments_array)
	{
		static const int tile_size = 32;
		concurrency::parallel_for_each(acc_view, moments_array.get_extent(), [=](concurrency::index<1> idx) restrict(amp) {
			moments_array(idx).clear();
		});
		concurrency::parallel_for_each(label_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if (label_array.get_extent().contains(idx.global))
			{
				int label = label_array(idx.global);
				moments& gm = moments_array(label);
				unsigned int y = idx.global[0];
				unsigned int x = idx.global[1];
				concurrency::atomic_fetch_inc(&gm.m00);
				concurrency::atomic_fetch_add(&gm.m01, y);
				concurrency::atomic_fetch_add(&gm.m10, x);
				concurrency::atomic_fetch_add(&gm.m02, y * y);
				concurrency::atomic_fetch_add(&gm.m11, x * y);
				concurrency::atomic_fetch_add(&gm.m20, x * x);
			}
		});
	}
}
