#pragma once

#include "amp_core.h"
#include "amp_calc_hist.h"
#include "amp_lut.h"

namespace amp
{
	inline void equalize_hist_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array)
	{
		concurrency::array<int, 1> hist(256, acc_view);
		concurrency::array<float, 1> lut(256, acc_view);
		calc_hist_32f_c1(acc_view, src_array, hist, 256);
		static const int tile_size = 256;
		int total = src_array.get_extent()[0] * src_array.get_extent()[1];
		parallel_for_each(acc_view, lut.get_extent().tile<tile_size>().pad(), [=, &hist, &lut](tiled_index<tile_size> idx) restrict(amp)
		{
			int lid = idx.local[0];
			tile_static int sumhist[256];
			tile_static float scale;
			sumhist[lid] = hist[lid];
			idx.barrier.wait_with_tile_static_memory_fence();

			if(lid == 0)
			{
				int sum = 0, i = 0;
				while(!sumhist[i]) ++i;

				if(total == sumhist[i])
				{
					scale = 1.0f;
					sumhist[i] = i;
				}
				else
				{
					scale = 255.0f / (total - sumhist[i]);

					for(sumhist[i] = 0; i < 256; i += 8)
					{
						sum += sumhist[i + 1];
						sumhist[i + 1] = sum;
						sum += sumhist[i + 2];
						sumhist[i + 2] = sum;
						sum += sumhist[i + 3];
						sumhist[i + 3] = sum;
						sum += sumhist[i + 4];
						sumhist[i + 4] = sum;
						sum += sumhist[i + 5];
						sumhist[i + 5] = sum;
						sum += sumhist[i + 6];
						sumhist[i + 6] = sum;
						sum += sumhist[i + 7];
						sumhist[i + 7] = sum;
						sum += sumhist[i + 8];
						sumhist[i + 8] = sum;
					}
				}
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			lut[lid] = float(int(sumhist[lid] * scale + 0.5f));
		});
		dest_array.discard_data();
		lut_32f_c1(acc_view, src_array, lut, dest_array);
	}
}
