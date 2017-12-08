#pragma once

#include "amp_core.h"
#include "amp_calc_hist.h"
#include "amp_lut.h"

namespace amp
{
	// Histogram Matching
	template<typename target_hist_type>
	inline void match_hist_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, const std::vector<target_hist_type>& target_hist)
	{
		int hist_size = int(target_hist.size());
		// 1.calculate histogram
		concurrency::array<int, 1> hist(hist_size, acc_view);
		array_view<int, 1> hist_view(hist);
		calc_hist_32f_c1(acc_view, src_array, hist_view, hist_size);
		// 2.download
		std::vector<int> cpu_hist(hist_size);
		copy(hist, cpu_hist.begin());
		// 3.normalize source and target hist
		std::vector<float> normalized_src_hist(hist_size);
		std::vector<float> normalized_target_hist(hist_size);
		target_hist_type total_target_hist = std::accumulate(target_hist.begin(), target_hist.end(), target_hist_type(0));
		for(int i = 0; i < hist_size; i++)
		{
			normalized_src_hist[i] = float(cpu_hist[i]) / src_array.get_extent().size();
			normalized_target_hist[i] = float(target_hist[i]) / total_target_hist;
		}
		for(int i = 1; i < hist_size; i++)
		{
			normalized_src_hist[i] += normalized_src_hist[i - 1];
			normalized_target_hist[i] += normalized_target_hist[i - 1];
		}
		// 4.calculate lut
		std::vector<float> cpu_lut(hist_size);
		int target_index = 0;
		for(int i = 0; i < hist_size; i++)
		{
			while(target_index < hist_size - 1 && (normalized_target_hist[target_index + 1] < normalized_src_hist[i]
				|| normalized_src_hist[i] - normalized_target_hist[target_index] > normalized_target_hist[target_index + 1] - normalized_src_hist[i]))
			{
				target_index++;
			}
			cpu_lut[i] = float(target_index);
		}
		// 5.apply lut
		concurrency::array<float, 1> lut(hist_size, acc_view);
		copy(cpu_lut.begin(), cpu_lut.end(), lut);
		lut_32f_c1(acc_view, src_array, lut, dest_array);
	}
}
