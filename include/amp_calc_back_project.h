#pragma once

#include "amp_core.h"
#include "amp_lut.h"

namespace amp
{
	inline void calc_back_project_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<const int, 1> hist, float scale = 1.0f)
	{
		const float rounding_eps = 0.000001f;
		concurrency::array<float, 1> lut(256, acc_view);
		int hist_bins = hist.get_extent()[0];
		parallel_for_each(acc_view, lut.get_extent(), [=, &lut](concurrency::index<1> idx) restrict(amp)
		{
			int x = idx[0];
			float value = float(x);
			float lb = 0.0f, ub = 256.0f, gap = (ub - lb) / hist_bins;
			value -= lb;
			int bin = int(value / gap + rounding_eps);
			if(bin >= hist_bins)
			{
				lut[x] = 0.0f;
			}
			else
			{
				lut[x] = fast_math::roundf(hist[bin] * scale);
			}
		});
		dest_array.discard_data();
		lut_32f_c1(acc_view, src_array, lut, dest_array);
	}
}
