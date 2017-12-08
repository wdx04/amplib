#pragma once

#include "amp_core.h"
#include "amp_conv2d.h"

namespace amp
{
	// Approximate Laplacian
	// kernel 0: 0, 1, 0, 1, -4, 1, 0, 1, 0
	// kernel 1: 2, 0, 2, 0, -8, 0, 2, 0, 2
	// kernel 2: -1, -3, -4, -3, -1, -3, 0, 6, 0, -3, -4, 6, 20, 6, -4, -3, 0, 6, 0, -3, -1, -3, -4, -3, -1
	// kernel 3: -2, -4, -4, -4, -2, -4, 0, 8, 0, -4, -4, 8, 24, 8, -4, -4, 0, 8, 0, -4, -2, -4, -4, -4, -2
	inline void laplacian_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, int kernel_index)
	{
		if(kernel_index == 0)
		{
			kernel_wrapper<float, 169U> k(3, 3, { 0, 1, 0, 1, -4, 1, 0, 1, 0 });
			convolve2d_32f_c1(acc_view, src_array, dest_array, k);
		}
		else if(kernel_index == 1)
		{
			kernel_wrapper<float, 169U> k(3, 3, { 2, 0, 2, 0, -8, 0, 2, 0, 2 });
			convolve2d_32f_c1(acc_view, src_array, dest_array, k);
		}
		else if(kernel_index == 2)
		{
			kernel_wrapper<float, 169U> k(5, 5, { -1, -3, -4, -3, -1, -3, 0, 6, 0, -3, -4, 6, 20, 6, -4, -3, 0, 6, 0, -3, -1, -3, -4, -3, -1 });
			convolve2d_32f_c1(acc_view, src_array, dest_array, k);
		}
		else if(kernel_index == 3)
		{
			kernel_wrapper<float, 169U> k(5, 5, { -2, -4, -4, -4, -2, -4, 0, 8, 0, -4, -4, 8, 24, 8, -4, -4, 0, 8, 0, -4, -2, -4, -4, -4, -2 });
			convolve2d_32f_c1(acc_view, src_array, dest_array, k);
		}
	}
}
