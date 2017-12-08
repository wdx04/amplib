#pragma once

#include "amp_core.h"
#include "amp_max_min.h"
#include "amp_mean_stddev.h"
#include "amp_calc_hist.h"

namespace amp
{
	// Scaling
	inline void scale_to_range_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array, float gain = 255.0f, float saturation = 0.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> max_min = max_min_value_32f_c1(acc_view, src_array, temp_array);
		if(max_min.first == max_min.second)
		{
			// fill 0.0f
			concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(dest_array, idx.global, 0.0f);
			});
		}
		else
		{
			// scale
			float alpha = gain / (max_min.first - max_min.second);
			float beta = max_min.second * gain / (max_min.second - max_min.first);
			int ignore_count = cvRound(src_array.get_extent().size() * saturation);
			if(ignore_count <= 0)
			{
				// no need to rescale
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(dest_array, idx.global, src_value * alpha + beta);
				});
			}
			else
			{
				// scale to temp array
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(temp_array, idx.global, src_value * alpha + beta);
				});
				int bin_size = cvRound(gain);
				std::vector<int> host_hist(bin_size);
				array_view<int, 1> device_hist(bin_size, host_hist);
				device_hist.discard_data();
				calc_hist_32f_c1(acc_view, temp_array, device_hist, bin_size);
				device_hist.synchronize();
				int sum_low = 0;
				int sum_high = 0;
				auto it_low = std::find_if(host_hist.begin(), host_hist.end(), [ignore_count, &sum_low](int cnt) -> bool
				{
					sum_low += cnt;
					return sum_low > ignore_count;
				});
				auto it_high = std::find_if(host_hist.rbegin(), host_hist.rend(), [ignore_count, &sum_high](int cnt) -> bool
				{
					sum_high += cnt;
					return sum_high > ignore_count;
				});
				float new_min = (float(it_low - host_hist.begin()) - beta) / alpha;
				float new_max = ((gain - float(it_high - host_hist.rbegin())) - beta) / alpha;
				alpha = gain / (new_max - new_min);
				beta = new_min * gain / (new_min - new_max);
				// rescale to dest array
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(dest_array, idx.global, src_value * alpha + beta);
				});
			}
		}
	}

	inline void scale_to_range_32f_c1(accelerator_view& acc_view, array_view<float, 2> src_dest_array, array_view<float, 2> temp_array, float gain = 255.0f, float saturation = 0.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> max_min = max_min_value_32f_c1(acc_view, src_dest_array, temp_array);
		if(max_min.first == max_min.second)
		{
			// fill 0.0f
			concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(src_dest_array, idx.global, 0.0f);
			});
		}
		else
		{
			// scale
			float alpha = gain / (max_min.first - max_min.second);
			float beta = max_min.second * gain / (max_min.second - max_min.first);
			int ignore_count = cvRound(src_dest_array.get_extent().size() * saturation);
			if(ignore_count <= 0)
			{
				// no need to rescale
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(src_dest_array, idx.global, src_value * alpha + beta);
				});
			}
			else
			{
				// scale to temp array
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(temp_array, idx.global, src_value * alpha + beta);
				});
				int bin_size = cvRound(gain);
				std::vector<int> host_hist(bin_size);
				array_view<int, 1> device_hist(bin_size, host_hist);
				device_hist.discard_data();
				calc_hist_32f_c1(acc_view, temp_array, device_hist, bin_size);
				device_hist.synchronize();
				int sum_low = 0;
				int sum_high = 0;
				auto it_low = std::find_if(host_hist.begin(), host_hist.end(), [ignore_count, &sum_low](int cnt) -> bool
				{
					sum_low += cnt;
					return sum_low > ignore_count;
				});
				auto it_high = std::find_if(host_hist.rbegin(), host_hist.rend(), [ignore_count, &sum_high](int cnt) -> bool
				{
					sum_high += cnt;
					return sum_high > ignore_count;
				});
				float new_min = (float(it_low - host_hist.begin()) - beta) / alpha;
				float new_max = ((gain - float(it_high - host_hist.rbegin())) - beta) / alpha;
				alpha = gain / (new_max - new_min);
				beta = new_min * gain / (new_min - new_max);
				// rescale to dest array
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(src_dest_array, idx.global, src_value * alpha + beta);
				});
			}
		}
	}

	inline void scale_to_range_abs_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> temp_array, float gain = 255.0f, float saturation = 0.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> max_min = abs_max_min_value_32f_c1(acc_view, src_array, temp_array);
		if(max_min.first == max_min.second)
		{
			// fill 0.0f
			concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(dest_array, idx.global, 0.0f);
			});
		}
		else
		{
			// scale
			float alpha = gain / (max_min.first - max_min.second);
			float beta = max_min.second * gain / (max_min.second - max_min.first);
			int ignore_count = cvRound(src_array.get_extent().size() * saturation);
			if(ignore_count <= 0)
			{
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(dest_array, idx.global, fast_math::fabsf(src_value) * alpha + beta);
				});
			}
			else
			{
				// scale to temp array
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(temp_array, idx.global, fast_math::fabsf(src_value) * alpha);
				});
				int bin_size = cvRound(gain);
				std::vector<int> host_hist(bin_size);
				array_view<int, 1> device_hist(bin_size, host_hist);
				device_hist.discard_data();
				calc_hist_32f_c1(acc_view, temp_array, device_hist, bin_size);
				device_hist.synchronize();
				int sum_low = 0;
				int sum_high = 0;
				auto it_low = std::find_if(host_hist.begin(), host_hist.end(), [ignore_count, &sum_low](int cnt) -> bool
				{
					sum_low += cnt;
					return sum_low > ignore_count;
				});
				auto it_high = std::find_if(host_hist.rbegin(), host_hist.rend(), [ignore_count, &sum_high](int cnt) -> bool
				{
					sum_high += cnt;
					return sum_high > ignore_count;
				});
				float new_min = float(it_low - host_hist.begin()) / alpha;
				float new_max = (gain - float(it_high - host_hist.rbegin())) / alpha;
				alpha = gain / (new_max - new_min);
				float beta = new_min * gain / (new_min - new_max);
				concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_array, idx.global);
					guarded_write(dest_array, idx.global, fast_math::fabsf(src_value) * alpha + beta);
				});
			}
		}
	}

	inline void scale_to_range_abs_32f_c1(accelerator_view& acc_view, array_view<float, 2> src_dest_array, array_view<float, 2> temp_array, float gain = 255.0f, float saturation = 0.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> max_min = abs_max_min_value_32f_c1(acc_view, src_dest_array, temp_array);
		if(max_min.first == max_min.second)
		{
			// fill 0.0f
			concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(src_dest_array, idx.global, 0.0f);
			});
		}
		else
		{
			// scale
			float alpha = gain / (max_min.first - max_min.second);
			float beta = max_min.second * gain / (max_min.second - max_min.first);
			int ignore_count = cvRound(src_dest_array.get_extent().size() * saturation);
			if(ignore_count <= 0)
			{
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(src_dest_array, idx.global, fast_math::fabsf(src_value) * alpha + beta);
				});
			}
			else
			{
				// scale to temp array
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(temp_array, idx.global, fast_math::fabsf(src_value) * alpha);
				});
				int bin_size = cvRound(gain);
				std::vector<int> host_hist(bin_size);
				array_view<int, 1> device_hist(bin_size, host_hist);
				device_hist.discard_data();
				calc_hist_32f_c1(acc_view, temp_array, device_hist, bin_size);
				device_hist.synchronize();
				int sum_low = 0;
				int sum_high = 0;
				auto it_low = std::find_if(host_hist.begin(), host_hist.end(), [ignore_count, &sum_low](int cnt) -> bool
				{
					sum_low += cnt;
					return sum_low > ignore_count;
				});
				auto it_high = std::find_if(host_hist.rbegin(), host_hist.rend(), [ignore_count, &sum_high](int cnt) -> bool
				{
					sum_high += cnt;
					return sum_high > ignore_count;
				});
				float new_min = float(it_low - host_hist.begin()) / alpha;
				float new_max = (gain - float(it_high - host_hist.rbegin())) / alpha;
				alpha = gain / (new_max - new_min);
				float beta = new_min * gain / (new_min - new_max);
				concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					float src_value = guarded_read(src_dest_array, idx.global);
					guarded_write(src_dest_array, idx.global, fast_math::fabsf(src_value) * alpha + beta);
				});
			}
		}
	}

	inline void scale_by_stddev_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, float gain = 255.0f, float dynamic = 3.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> mean_stddev = mean_std_dev_32f_c1(acc_view, src_array);
		float mean_val = mean_stddev.first;
		float stddev_val = mean_stddev.second;
		float max_val = mean_val + dynamic * stddev_val;
		float min_val = mean_val - dynamic * stddev_val;
		if(max_val <= min_val)
		{
			// fill with mean value
			concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(dest_array, idx.global, mean_val);
			});
			return;
		}
		// scale
		float alpha = gain / (max_val - min_val);
		float beta = min_val * gain / (min_val - max_val);
		// no need to rescale
		concurrency::parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float src_value = guarded_read(src_array, idx.global);
			guarded_write(dest_array, idx.global, src_value * alpha + beta);
		});
	}

	inline void scale_by_stddev_32f_c1(accelerator_view& acc_view, array_view<float, 2> src_dest_array, float gain = 255.0f, float dynamic = 3.0f)
	{
		static const int tile_size = 32;
		std::pair<float, float> mean_stddev = mean_std_dev_32f_c1(acc_view, src_dest_array);
		float mean_val = mean_stddev.first;
		float stddev_val = mean_stddev.second;
		float max_val = mean_val + dynamic * stddev_val;
		float min_val = mean_val - dynamic * stddev_val;
		if(max_val <= min_val)
		{
			// fill with mean value
			concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(src_dest_array, idx.global, mean_val);
			});
			return;
		}
		// scale
		float alpha = gain / (max_val - min_val);
		float beta = min_val * gain / (min_val - max_val);
		// no need to rescale
		concurrency::parallel_for_each(acc_view, src_dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float src_value = guarded_read(src_dest_array, idx.global);
			guarded_write(src_dest_array, idx.global, src_value * alpha + beta);
		});
	}

}
