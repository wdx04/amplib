#pragma once

#include "amp_core.h"
#include "amp_mean_stddev.h"
#include "amp_gaussian.h"
#include "amp_convert_color.h"
#include "amp_color_diff.h"
#include "amp_scale.h"

namespace amp
{
	// FT Salient Region Detection(L*a*b* color space only)
	inline void ft_salient_region_detection(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_array, array_view<float, 2> temp_array1
		, array_view<float, 2> temp_array2, array_view<float, 2> temp_array3, float gaussian_sigma = 1.0f, float saturation = 0.0f)
	{
		float mean_ch1 = mean_std_dev_32f_c1(acc_view, src_channel1).first;
		float mean_ch2 = mean_std_dev_32f_c1(acc_view, src_channel2).first;
		float mean_ch3 = mean_std_dev_32f_c1(acc_view, src_channel3).first;
		gaussian_filter_32f_c1(acc_view, src_channel1, temp_array1, dest_array, 0, gaussian_sigma);
		gaussian_filter_32f_c1(acc_view, src_channel2, temp_array2, dest_array, 0, gaussian_sigma);
		gaussian_filter_32f_c1(acc_view, src_channel3, temp_array3, dest_array, 0, gaussian_sigma);
		static const int tile_size = 32;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if(dest_array.get_extent().contains(gidx))
			{
				float_3 src_value(temp_array1(gidx), temp_array2(gidx), temp_array3(gidx));
				float_3 mean_value(mean_ch1, mean_ch2, mean_ch3);
				dest_array(gidx) = ciede1976_delta_e(src_value, mean_value);
			}
		});
		scale_to_range_32f_c1(acc_view, dest_array, temp_array1, 255.0f, saturation);
	}
}
