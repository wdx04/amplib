#pragma once

#include "amp_core.h"
#include "amp_erode.h"
#include "amp_dilate.h"

namespace amp
{
	enum class morph_gradient_type { combined, internal, external };
	inline void morph_gradient_32f_c1(accelerator_view acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, const cv::Mat& kernel, morph_gradient_type type = morph_gradient_type::combined)
	{
		static const int tile_size = 32;
		kernel_wrapper<float, 169U> warpped_kernel(kernel);
		if(type == morph_gradient_type::combined)
		{
			erode_32f_c1(acc_view, src_array, dest_array, warpped_kernel);
			dilate_32f_c1(acc_view, src_array, temp_array, warpped_kernel);
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				if(dest_array.get_extent().contains(gidx))
				{
					dest_array(gidx) = temp_array(gidx) - dest_array(gidx);
				}
			});
		}
		else if(type == morph_gradient_type::external)
		{
			dilate_32f_c1(acc_view, src_array, dest_array, warpped_kernel);
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				if(dest_array.get_extent().contains(gidx))
				{
					dest_array(gidx) = dest_array(gidx) - src_array(gidx);
				}
			});
		}
		else if(type == morph_gradient_type::internal)
		{
			erode_32f_c1(acc_view, src_array, dest_array, warpped_kernel);
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				if(dest_array.get_extent().contains(gidx))
				{
					dest_array(gidx) = src_array(gidx) - dest_array(gidx);
				}
			});
		}
	}
}
