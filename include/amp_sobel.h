#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include "amp_core.h"
#include "amp_conv_separable.h"

namespace amp
{
	// Sobel Filter
	template<unsigned int max_kernel_size = 1024U>
	inline void sobel_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> row_temp_array
		, int dx, int dy, int ksize)
	{
		cv::Mat kx, ky;
		cv::getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);
		convolve_separable_32f_c1<max_kernel_size>(acc_view, src_array, dest_array, row_temp_array, kx, ky);
	}
}
