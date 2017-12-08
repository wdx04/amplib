#pragma once

#include "amp_core.h"
#include "amp_dilate.h"
#include "amp_geodesic_erode.h"

namespace amp
{
	// Geodesic close
	inline void geodesic_close_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, const cv::Mat& kernel, int dilate_iter = 1, int max_erode_iter = 0)
	{
		// 1.dilate
		dilate_32f_c1(acc_view, src_array, temp_array, dest_array, kernel, dilate_iter);
		// 2.reconstruct
		geodesic_erode_32f_c1(acc_view, temp_array, dest_array, src_array, temp_array, max_erode_iter);
	}
}
