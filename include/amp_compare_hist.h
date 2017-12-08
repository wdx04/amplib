#pragma once

#include "amp_core.h"

namespace amp
{
	enum class compare_hist_method { correlation = 0, chebyshev = 1, hellinger = 2 };
	inline void compare_hist_32f_c1(accelerator_view& acc_view, array_view<const int, 2> array1, array_view<const int, 2> array2
		, array_view<float, 2> compare_results, compare_hist_method method = compare_hist_method::correlation)
	{
		assert(array1.get_extent()[1] == array2.get_extent()[1]);
		assert(array1.get_extent()[0] == compare_results.get_extent()[0]);
		assert(array2.get_extent()[0] == compare_results.get_extent()[1]);
		static const int max_bin_size = 256;
		int numBins = array1.get_extent()[1];
		if(method == compare_hist_method::correlation)
		{
			std::vector<int> sample_hist(numBins);
			concurrency::copy(array1[0], sample_hist.begin());
			int nPixCount = std::accumulate(sample_hist.begin(), sample_hist.end(), 0);
			float fHistAvg = float(nPixCount) / numBins;
			parallel_for_each(acc_view, compare_results.get_extent(), [=](concurrency::index<2> idx) restrict(amp)
			{
				int templateIdx = idx[0];
				int imageIdx = idx[1];
				float sumP1 = 0.0f, sumP2 = 0.0f, sumP3 = 0.0f;
				for(int i = 0; i < numBins; i++)
				{
					float templateVal = float(array1(templateIdx, i));
					float imageVal = float(array2(imageIdx, i));
					sumP1 += (imageVal - fHistAvg) * (templateVal - fHistAvg);
					sumP2 += (imageVal - fHistAvg) * (imageVal - fHistAvg);
					sumP3 += (templateVal - fHistAvg) * (templateVal - fHistAvg);
				}
				compare_results[idx] = sumP1 / fast_math::sqrtf(sumP2 * sumP3);
			});
		}
		else if(method == compare_hist_method::chebyshev)
		{
			parallel_for_each(acc_view, compare_results.get_extent(), [=](concurrency::index<2> idx) restrict(amp)
			{
				int templateIdx = idx[0];
				int imageIdx = idx[1];
				float sum = 0.0f;
				for(int i = 0; i < numBins; i++)
				{
					float templateVal = float(array1(templateIdx, i));
					float imageVal = float(array2(imageIdx, i));
					sum += fast_math::fabsf(templateVal - imageVal);
				}
				compare_results[idx] = sum;
			});
		}
		else if(method == compare_hist_method::hellinger)
		{
			std::vector<int> sample_hist(numBins);
			concurrency::copy(array1[0], sample_hist.begin());
			int nPixCount = std::accumulate(sample_hist.begin(), sample_hist.end(), 0);
			float fHistAvg = float(nPixCount) / numBins;
			parallel_for_each(acc_view, compare_results.get_extent(), [=](concurrency::index<2> idx) restrict(amp)
			{
				int templateIdx = idx[0];
				int imageIdx = idx[1];
				float sum = 0.0f;
				for(int i = 0; i < numBins; i++)
				{
					float templateVal = float(array1(templateIdx, i));
					float imageVal = float(array2(imageIdx, i));
					sum += fast_math::sqrtf(templateVal * imageVal);
				}
				compare_results[idx] = fast_math::sqrtf(1.0f - sum / (fHistAvg * numBins));
			});
		}
	}
}
