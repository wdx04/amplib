#pragma once

#include "amp_core.h"

namespace amp
{
	// Max and Mean value
	inline std::pair<float, float> max_min_value_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> temp_array)
	{
		std::pair<float, float> result;
		// step1: reduce 2d to 1d
		int src_cols = src_array.get_extent()[1];
		int src_rows = src_array.get_extent()[0];
		parallel_for_each(acc_view, concurrency::extent<1>(src_cols), [=](concurrency::index<1> idx) restrict(amp)
		{
			int global_col = idx[0];
			float max_value = -FLT_MAX;
			float min_value = FLT_MAX;
			for(int i = 0; i < src_rows; i++)
			{
				float src_value = src_array(i, global_col);
				max_value = fast_math::fmaxf(max_value, src_value);
				min_value = fast_math::fminf(min_value, src_value);
			}
			temp_array(0, global_col) = max_value;
			temp_array(1, global_col) = min_value;
		});
		// step2: reduce 1d to single value
		std::vector<float> cpu_temp(src_cols * 2);
		concurrency::copy(temp_array.section(0, 0, 2, src_cols), cpu_temp.begin());
		result.first = *std::max_element(cpu_temp.begin(), cpu_temp.begin() + src_cols);
		result.second = *std::min_element(cpu_temp.begin() + src_cols, cpu_temp.end());
		return result;
	}

	inline std::pair<float, float> max_min_value_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array)
	{
		std::pair<float, float> result;
		concurrency::array<float, 2> temp_array(2, src_array.get_extent()[1], acc_view);
		// step1: reduce 2d to 1d
		int src_cols = src_array.get_extent()[1];
		int src_rows = src_array.get_extent()[0];
		parallel_for_each(acc_view, concurrency::extent<1>(src_cols), [=, &temp_array](concurrency::index<1> idx) restrict(amp)
		{
			int global_col = idx[0];
			float max_value = -FLT_MAX;
			float min_value = FLT_MAX;
			for(int i = 0; i < src_rows; i++)
			{
				float src_value = src_array(i, global_col);
				max_value = fast_math::fmaxf(max_value, src_value);
				min_value = fast_math::fminf(min_value, src_value);
			}
			temp_array(0, global_col) = max_value;
			temp_array(1, global_col) = min_value;
		});
		// step2: reduce 1d to single value
		std::vector<float> cpu_temp(src_cols * 2);
		concurrency::copy(temp_array.section(0, 0, 2, src_cols), cpu_temp.begin());
		result.first = *std::max_element(cpu_temp.begin(), cpu_temp.begin() + src_cols);
		result.second = *std::min_element(cpu_temp.begin() + src_cols, cpu_temp.end());
		return result;
	}

	inline std::pair<float, float> abs_max_min_value_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> temp_array)
	{
		std::pair<float, float> result;
		// step1: reduce 2d to 1d
		int src_cols = src_array.get_extent()[1];
		int src_rows = src_array.get_extent()[0];
		parallel_for_each(acc_view, concurrency::extent<1>(src_cols), [=](concurrency::index<1> idx) restrict(amp)
		{
			int global_col = idx[0];
			float max_value = 0.0f;
			float min_value = FLT_MAX;
			for(int i = 0; i < src_rows; i++)
			{
				float src_value = fast_math::fabsf(src_array(i, global_col));
				max_value = fast_math::fmaxf(max_value, src_value);
				min_value = fast_math::fminf(min_value, src_value);
			}
			temp_array(0, global_col) = max_value;
			temp_array(1, global_col) = min_value;
		});
		// step2: reduce 1d to single value
		std::vector<float> cpu_temp(src_cols * 2);
		concurrency::copy(temp_array.section(0, 0, 2, src_cols), cpu_temp.begin());
		result.first = *std::max_element(cpu_temp.begin(), cpu_temp.begin() + src_cols);
		result.second = *std::min_element(cpu_temp.begin() + src_cols, cpu_temp.end());
		return result;
	}

	inline std::pair<float, float> abs_max_min_value_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array)
	{
		std::pair<float, float> result;
		concurrency::array<float, 2> temp_array(2, src_array.get_extent()[1], acc_view);
		// step1: reduce 2d to 1d
		int src_cols = src_array.get_extent()[1];
		int src_rows = src_array.get_extent()[0];
		parallel_for_each(acc_view, concurrency::extent<1>(src_cols), [=, &temp_array](concurrency::index<1> idx) restrict(amp)
		{
			int global_col = idx[0];
			float max_value = 0.0f;
			float min_value = FLT_MAX;
			for(int i = 0; i < src_rows; i++)
			{
				float src_value = fast_math::fabsf(src_array(i, global_col));
				max_value = fast_math::fmaxf(max_value, src_value);
				min_value = fast_math::fminf(min_value, src_value);
			}
			temp_array(0, global_col) = max_value;
			temp_array(1, global_col) = min_value;
		});
		// step2: reduce 1d to single value
		std::vector<float> cpu_temp(src_cols * 2);
		concurrency::copy(temp_array.section(0, 0, 2, src_cols), cpu_temp.begin());
		result.first = *std::max_element(cpu_temp.begin(), cpu_temp.begin() + src_cols);
		result.second = *std::min_element(cpu_temp.begin() + src_cols, cpu_temp.end());
		return result;
	}

}
