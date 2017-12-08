#pragma once

#include "amp_core.h"
#include "amp_erode.h"
#include "amp_dilate.h"

namespace amp
{
	// Open
	inline void open_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array
		, const cv::Mat& kernel, int iterations = 1)
	{
		erode_32f_c1(acc_view, src_array, temp_array, dest_array, kernel, iterations);
		dilate_32f_c1(acc_view, temp_array, dest_array, temp_array, kernel, iterations);
	}
}
