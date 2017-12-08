#pragma once

#include "amp_core.h"
#include "amp_geodesic_dilate.h"
#include "amp_geodesic_erode.h"

namespace amp
{
	// Flatten low-contrast areas in an image(preprocess for regional maxima/minima)
	void remove_low_contrast_basin(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array, float h, bool connective_4 = false)
	{
		static const int tile_size = 32;
		// plus h
		concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(src_array, idx.global) + h);
		});
		// reconstruct
		geodesic_erode_32f_c1(acc_view, dest_array, src_array, temp_array, 0, connective_4);
	}

	inline void regional_maxima_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array1
		, array_view<float, 2> temp_array2, bool connective_4 = false)
	{
		static const int tile_size = 32;
		// plus 1.0
		concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(src_array, idx.global) + 1.0f);
		});
		// reconstruct
		geodesic_dilate_32f_c1(acc_view, src_array, temp_array1, dest_array, temp_array2, 0, connective_4);
		concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float f = guarded_read(src_array, idx.global);
			float g = guarded_read(dest_array, idx.global) - guarded_read(temp_array1, idx.global);
			guarded_write(dest_array, idx.global, (g >= 1.0f || f >= 255.0f ? 255.0f : 0.0f));
		});
	}

	inline void regional_minima_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array1
		, array_view<float, 2> temp_array2, bool connective_4 = false)
	{
		static const int tile_size = 32;
		// minus 1.0
		concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(src_array, idx.global) - 1.0f);
		});
		// reconstruct
		geodesic_erode_32f_c1(acc_view, src_array, temp_array1, dest_array, temp_array2, 0, connective_4);
		concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float f = guarded_read(src_array, idx.global);
			float g = guarded_read(temp_array1, idx.global) - guarded_read(dest_array, idx.global);
			guarded_write(dest_array, idx.global, (g >= 1.0f || f == 0.0f ? 255.0f : 0.0f));
		});
	}

	// Image contrast enhancement or classification by the toggle operator.
	inline void toggle_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<const float, 2> lower_mask_array
		, array_view<const float, 2> higher_mask_array, bool is_binary_output = false)
	{
		static const int tile_size = 32;
		if (is_binary_output)
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, idx.global);
				float lower_value = guarded_read(lower_mask_array, idx.global);
				float upper_value = guarded_read(higher_mask_array, idx.global);
				float lower_diff = src_value - lower_value;
				float upper_diff = upper_value - src_value;
				guarded_write(dest_array, idx.global, (upper_diff <= lower_diff ? 255.0f : 0.0f));
			});
		}
		else
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, idx.global);
				float lower_value = guarded_read(lower_mask_array, idx.global);
				float upper_value = guarded_read(higher_mask_array, idx.global);
				float lower_diff = src_value - lower_value;
				float upper_diff = upper_value - src_value;
				guarded_write(dest_array, idx.global, (upper_diff <= lower_diff ? upper_value : lower_value));
			});
		}
	}

}
