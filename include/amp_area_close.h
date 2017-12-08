#pragma once

#include "amp_core.h"
#include "amp_area_open.h"

namespace amp
{
	inline void binary_area_close_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<int, 2> label_array, int n)
	{
		binary_area_fill_32f_c1(acc_view, src_array, dest_array, label_array, n, 255.0f);
	}
}
