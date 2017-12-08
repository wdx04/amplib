#pragma once

#include "amp_core.h"
#include "amp_open.h"
#include "amp_geodesic_dilate.h"

namespace amp
{
	// Top Hat
	inline void top_hat_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, const cv::Mat& kernel, int iterations = 1)
	{
		static const int tile_size = 32;
		open_32f_c1(acc_view, src_array, dest_array, temp_array, kernel, iterations);
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if(dest_array.get_extent().contains(gidx))
			{
				dest_array(gidx) = src_array(gidx) - dest_array(gidx);
			}
		});
	}

	// Reconstruction Top Hat
	inline void reconstruction_top_hat_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, const cv::Mat& kernel, int iterations = 1)
	{
		static const int tile_size = 32;
		erode_32f_c1(acc_view, src_array, dest_array, temp_array, kernel, iterations);
		geodesic_dilate_32f_c1(acc_view, dest_array, src_array, temp_array, 0, true);
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if (dest_array.get_extent().contains(gidx))
			{
				dest_array(gidx) = src_array(gidx) - dest_array(gidx);
			}
		});
	}

}
