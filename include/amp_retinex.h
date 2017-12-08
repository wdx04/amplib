#pragma once

#include "amp_core.h"
#include "amp_gaussian.h"
#include "amp_scale.h"

namespace amp
{
	// Retinex Filter
	enum class retinex_rescale_method { no_rescale = 0, simple_balance, mean_stddev_balance };
	inline void retinex_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array1, array_view<float, 2> temp_array2, int ksize, float sigma
		, retinex_rescale_method rescale_method = retinex_rescale_method::simple_balance, float rescale_param = 0.01f)
	{
		static const int tile_size = 32;
		// (1) source matrix +1.0f
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float src_val = guarded_read(src_array, idx.global);
			guarded_write(temp_array1, idx.global, src_val + 1.0f);
		});
		// (2) gaussian blur
		gaussian_filter_32f_c1(acc_view, temp_array1, temp_array2, dest_array, ksize, sigma);
		// (3) subtract
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float src_val1 = guarded_read(temp_array1, idx.global, 1.0f);
			float src_val2 = guarded_read(temp_array2, idx.global, 1.0f);
			guarded_write(dest_array, idx.global, fast_math::logf(src_val1) - fast_math::logf(src_val2));
		});
		// (4) gain
		switch(rescale_method)
		{
		case retinex_rescale_method::simple_balance:
			scale_to_range_32f_c1(acc_view, dest_array, temp_array1, 255.0f, rescale_param);
			break;
		case retinex_rescale_method::mean_stddev_balance:
			scale_by_stddev_32f_c1(acc_view, dest_array, 255.0f, rescale_param);
			break;
		}
	}

	inline void retinex_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<float, 2> temp_array1, array_view<float, 2> temp_array2
		, const std::vector<int>& ksizes, const std::vector<float>& sigmas, const std::vector<float>& weights
		, retinex_rescale_method rescale_method = retinex_rescale_method::simple_balance, float rescale_param = 0.01f)
	{
		static const int tile_size = 32;
		assert(ksizes.size() == sigmas.size() && ksizes.size() == weights.size());
		// (1) normalize weights
		std::vector<float> normal_weights(weights.size());
		float total_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
		if(total_weights > 1.0f + FLT_EPSILON)
		{
			std::transform(weights.begin(), weights.end(), normal_weights.begin(), [=](float w) -> float { return w / total_weights; });
		}
		else
		{
			std::copy(weights.begin(), weights.end(), normal_weights.begin());
		}
		// (1) source matrix +1.0f
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float target_val = guarded_read(src_array, idx.global) + 1.0f;
			guarded_write(temp_array1, idx.global, target_val);
			guarded_write(dest_array, idx.global, fast_math::logf(target_val));
		});
		for(size_t i = 0; i < weights.size(); i++)
		{
			int ksize = ksizes[i];
			float sigma = sigmas[i];
			float weight = normal_weights[i];
			// (2) gaussian blur
			gaussian_filter_32f_c1(acc_view, temp_array1, temp_array1, temp_array2, ksize, sigma);
			// (3) subtract
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_val1 = guarded_read(dest_array, idx.global);
				float src_val2 = fast_math::logf(guarded_read(temp_array1, idx.global, 1.0f));
				guarded_write(dest_array, idx.global, src_val1 - src_val2 * weight);
			});
		}
		// (4) gain
		switch(rescale_method)
		{
		case retinex_rescale_method::simple_balance:
			scale_to_range_32f_c1(acc_view, dest_array, temp_array1, 255.0f, rescale_param);
			break;
		case retinex_rescale_method::mean_stddev_balance:
			scale_by_stddev_32f_c1(acc_view, dest_array, 255.0f, rescale_param);
			break;
		}
	}

	inline void retinex_cr_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3, array_view<float, 2> temp_array1, array_view<float, 2> temp_array2
		, const std::vector<int>& ksizes, const std::vector<float>& sigmas, const std::vector<float>& weights
		, bool use_intensity_channel = false, retinex_rescale_method rescale_method = retinex_rescale_method::simple_balance, float rescale_param = 0.01f)
	{
		static const int tile_size = 32;
		if(use_intensity_channel)
		{
			parallel_for_each(acc_view, temp_array1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(temp_array1, idx.global, (guarded_read(src_channel1, idx.global)
					+ guarded_read(src_channel2, idx.global) + guarded_read(src_channel3, idx.global)) / 3.0f);
			});
			retinex_32f_c1(acc_view, temp_array1, temp_array2, dest_channel2, dest_channel3, ksizes, sigmas, weights, amp::retinex_rescale_method::no_rescale);
			switch(rescale_method)
			{
			case retinex_rescale_method::simple_balance:
				scale_to_range_32f_c1(acc_view, temp_array2, temp_array1, 255.0f, rescale_param);
				break;
			case retinex_rescale_method::mean_stddev_balance:
				scale_by_stddev_32f_c1(acc_view, temp_array2, 255.0f, rescale_param);
				break;
			}
			parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float gray = fast_math::fmaxf(guarded_read(temp_array1, idx.global), 1.0f);
				float gray1 = guarded_read(temp_array2, idx.global);
				float factor = fast_math::fminf(gray1 / gray, 3.0f);
				float channel1_in = guarded_read(src_channel1, idx.global);
				float channel2_in = guarded_read(src_channel2, idx.global);
				float channel3_in = guarded_read(src_channel3, idx.global);
				float channel1_out = factor * channel1_in;
				float channel2_out = factor * channel2_in;
				float channel3_out = factor * channel3_in;
				if(channel1_out > 255.0f || channel2_out > 255.0f || channel3_out > 255.0f)
				{
					factor = 255.0f / fast_math::fmaxf(fast_math::fmaxf(channel1_in, channel2_in), channel3_in);
					channel1_out = factor * channel1_in;
					channel2_out = factor * channel2_in;
					channel3_out = factor * channel3_in;
				}
				guarded_write(dest_channel1, idx.global, channel1_out);
				guarded_write(dest_channel2, idx.global, channel2_out);
				guarded_write(dest_channel3, idx.global, channel3_out);
			});
		}
		else
		{
			retinex_32f_c1(acc_view, src_channel1, dest_channel1, temp_array1, temp_array2, ksizes, sigmas, weights, amp::retinex_rescale_method::no_rescale);
			retinex_32f_c1(acc_view, src_channel2, dest_channel2, temp_array1, temp_array2, ksizes, sigmas, weights, amp::retinex_rescale_method::no_rescale);
			retinex_32f_c1(acc_view, src_channel3, dest_channel3, temp_array1, temp_array2, ksizes, sigmas, weights, amp::retinex_rescale_method::no_rescale);
			parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_ch1 = guarded_read(src_channel1, idx.global);
				float dest_ch1 = guarded_read(dest_channel1, idx.global);
				float src_ch2 = guarded_read(src_channel2, idx.global);
				float dest_ch2 = guarded_read(dest_channel2, idx.global);
				float src_ch3 = guarded_read(src_channel3, idx.global);
				float dest_ch3 = guarded_read(dest_channel3, idx.global);
				float logl = fast_math::logf(src_ch1 + src_ch2 + src_ch3 + 3.0f);
				guarded_write(dest_channel1, idx.global, (fast_math::logf(128.0f * (src_ch1 + 1.0f)) - logl) * dest_ch1);
				guarded_write(dest_channel2, idx.global, (fast_math::logf(128.0f * (src_ch2 + 1.0f)) - logl) * dest_ch2);
				guarded_write(dest_channel3, idx.global, (fast_math::logf(128.0f * (src_ch3 + 1.0f)) - logl) * dest_ch3);
			});
			switch(rescale_method)
			{
			case retinex_rescale_method::simple_balance:
				scale_to_range_32f_c1(acc_view, dest_channel1, temp_array1, 255.0f, rescale_param);
				scale_to_range_32f_c1(acc_view, dest_channel2, temp_array2, 255.0f, rescale_param);
				scale_to_range_32f_c1(acc_view, dest_channel3, temp_array1, 255.0f, rescale_param);
				break;
			case retinex_rescale_method::mean_stddev_balance:
				scale_by_stddev_32f_c1(acc_view, dest_channel1, 255.0f, rescale_param);
				scale_by_stddev_32f_c1(acc_view, dest_channel2, 255.0f, rescale_param);
				scale_by_stddev_32f_c1(acc_view, dest_channel3, 255.0f, rescale_param);
				break;
			}
		}
	}

}
