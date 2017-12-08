#pragma once

#include "amp_core.h"
#include "amp_erode.h"
#include "amp_geodesic_dilate.h"

namespace amp
{
	// Geodesic open
	inline void geodesic_open_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, const cv::Mat& kernel, int erode_iter = 1, int max_dilate_iter = 0)
	{
		// 1.erode
		erode_32f_c1(acc_view, src_array, temp_array, dest_array, kernel, erode_iter);
		// 2.reconstruct
		geodesic_dilate_32f_c1(acc_view, temp_array, dest_array, src_array, temp_array, max_dilate_iter);
	}

}
