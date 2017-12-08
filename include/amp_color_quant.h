#pragma once

#include "amp_core.h"
#include "amp_color_diff.h"

namespace amp
{
	inline void initialize_pallete_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float_3, 2> pallete)
	{
		// 1.calculate max/min values for each channel
		std::vector<float> result(6);
		concurrency::array<float, 2> temp_array(6, src_channel1.get_extent()[1], acc_view);
		// step1: reduce 2d to 1d
		int src_cols = src_channel1.get_extent()[1];
		int src_rows = src_channel1.get_extent()[0];
		parallel_for_each(acc_view, concurrency::extent<1>(src_cols), [=, &temp_array](concurrency::index<1> idx) restrict(amp)
		{
			int global_col = idx[0];
			float max_c1 = -FLT_MAX;
			float min_c1 = FLT_MAX;
			float max_c2 = -FLT_MAX;
			float min_c2 = FLT_MAX;
			float max_c3 = -FLT_MAX;
			float min_c3 = FLT_MAX;
			for (int i = 0; i < src_rows; i++)
			{
				float c1_value = src_channel1(i, global_col);
				max_c1 = fast_math::fmaxf(max_c1, c1_value);
				min_c1 = fast_math::fminf(min_c1, c1_value);
				float c2_value = src_channel2(i, global_col);
				max_c2 = fast_math::fmaxf(max_c2, c2_value);
				min_c2 = fast_math::fminf(min_c2, c2_value);
				float c3_value = src_channel3(i, global_col);
				max_c3 = fast_math::fmaxf(max_c3, c3_value);
				min_c3 = fast_math::fminf(min_c3, c3_value);
			}
			temp_array(0, global_col) = max_c1;
			temp_array(1, global_col) = min_c1;
			temp_array(2, global_col) = max_c2;
			temp_array(3, global_col) = min_c2;
			temp_array(4, global_col) = max_c3;
			temp_array(5, global_col) = min_c3;
		});
		// step2: reduce 1d to single value
		std::vector<float> cpu_temp(src_cols * 6);
		concurrency::copy(temp_array.section(0, 0, 6, src_cols), cpu_temp.begin());
		result[0] = *std::max_element(cpu_temp.begin(), cpu_temp.begin() + src_cols);
		result[1] = *std::min_element(cpu_temp.begin() + src_cols, cpu_temp.begin() + src_cols * 2);
		result[2] = *std::max_element(cpu_temp.begin() + src_cols * 2, cpu_temp.begin() + src_cols * 3);
		result[3] = *std::min_element(cpu_temp.begin() + src_cols * 3, cpu_temp.begin() + src_cols * 4);
		result[4] = *std::max_element(cpu_temp.begin() + src_cols * 4, cpu_temp.begin() + src_cols * 5);
		result[5] = *std::min_element(cpu_temp.begin() + src_cols * 5, cpu_temp.end());
		std::vector<cv::Vec3f> init_pallete;
		int pallete_size = pallete.get_extent()[0] * pallete.get_extent()[1];
		float step1 = (result[0] - result[1]) / (pallete_size - 1);
		float step2 = (result[2] - result[3]) / (pallete_size - 1);
		float step3 = (result[4] - result[5]) / (pallete_size - 1);
		for(int i = 0; i < pallete_size; i++)
		{
			float value1 = result[1] + step1 * i;
			float value2 = result[3] + step2 * i;
			float value3 = result[5] + step3 * i;
			init_pallete.push_back(cv::Vec3f(value1, value2, value3));
		}
		concurrency::copy((float_3*)&init_pallete[0], pallete);
	}

	template<int array_height, int array_width>
	inline void local_sum_up(int_4(&local_sums)[array_height][array_width], int i, int j, int x, int y, int z) restrict(amp)
	{
		bool in_range = i >= 0 && i < array_height && j >= 0 && j < array_width;
		if(in_range)
		{
			int_4& local_sum = local_sums[i][j];
			concurrency::atomic_fetch_add(&local_sum.ref_x(), x);
			concurrency::atomic_fetch_add(&local_sum.ref_y(), y);
			concurrency::atomic_fetch_add(&local_sum.ref_z(), z);
			concurrency::atomic_fetch_inc(&local_sum.ref_w());
		}
	}

	inline void generate_pallete_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float_3, 2> pallete, int iterations, bool use_8_neighbour = false, bool ciede00_delta = false)
	{
		static const int tile_size = 32;
		static const int max_pallete_height = 16;
		static const int max_pallete_width = 16;
		int pallete_height = pallete.get_extent()[0];
		int pallete_width = pallete.get_extent()[1];
		assert(pallete_height <= max_pallete_height);
		assert(pallete_width <= max_pallete_width);
		int pallete_size = pallete_height * pallete_width;
		concurrency::array<int_4, 2> pallete_sum(pallete.get_extent(), acc_view);
		concurrency::array_view<int_4, 2> pallete_sum_view(pallete_sum);
		for(int i = 0; i < iterations; i++)
		{
			// 1 calculate neighbourhood size
			int neighbour_size = 0;
			if(i < iterations / 8 && pallete_height >= 5 && pallete_width >= 5)
			{
				neighbour_size = 2;
			}
			else if(i < iterations / 2 && (pallete_height >= 3 && pallete_height >= 3 || !use_8_neighbour))
			{
				neighbour_size = 1;
			}
			// 2 set pallete sum to zero
			parallel_for_each(acc_view, pallete_sum_view.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(pallete_sum_view, idx.global, int_4());
			});
			// 3 recalculate pallete sum
			parallel_for_each(acc_view, src_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static int_4 local_sums[max_pallete_height][max_pallete_width + 1];
				if(idx.local[0] < pallete_height && idx.local[1] < pallete_width)
				{
					local_sums[idx.local[0]][idx.local[1]] = int_4();
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				concurrency::index<2> gidx = idx.global;
				if(src_channel1.get_extent().contains(gidx))
				{
					float_3 src_value(src_channel1(gidx), src_channel2(gidx), src_channel3(gidx));
					float min_sqdiff = FLT_MAX;
					concurrency::index<2> min_sqdiff_index(-1, -1);
					for(int j = 0; j < pallete_size; j++)
					{
						concurrency::index<2> pal_index(j / pallete_width, j % pallete_width);
						float_3 pallete_value = pallete(pal_index);
						float sqdiff = ciede00_delta ? ciede2000_delta_e(src_value, pallete_value) : ciede1976_delta_e(src_value, pallete_value);
						if(sqdiff < min_sqdiff)
						{
							min_sqdiff = sqdiff;
							min_sqdiff_index = pal_index;
						}
					}
					int x = int(fast_math::roundf(src_value.x));
					int y = int(fast_math::roundf(src_value.y));
					int z = int(fast_math::roundf(src_value.z));
					local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1], x, y, z);
					if(neighbour_size == 1)
					{
						if(use_8_neighbour)
						{
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] + 1, x, y, z);
						}
						else
						{
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1], x, y, z);
						}
					}
					else if(neighbour_size == 2)
					{
						if(use_8_neighbour)
						{
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1] + 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] + 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] + 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1] + 2, x, y, z);
						}
						else
						{
							local_sum_up(local_sums, min_sqdiff_index[0] - 2, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] - 1, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0], min_sqdiff_index[1] + 2, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] - 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1], x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 1, min_sqdiff_index[1] + 1, x, y, z);
							local_sum_up(local_sums, min_sqdiff_index[0] + 2, min_sqdiff_index[1], x, y, z);
						}
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				if(idx.local[0] < pallete_height && idx.local[1] < pallete_width)
				{
					int_4& global_sum = pallete_sum_view(idx.local);
					int_4& local_sum = local_sums[idx.local[0]][idx.local[1]];
					concurrency::atomic_fetch_add(&global_sum.ref_x(), local_sum.x);
					concurrency::atomic_fetch_add(&global_sum.ref_y(), local_sum.y);
					concurrency::atomic_fetch_add(&global_sum.ref_z(), local_sum.z);
					concurrency::atomic_fetch_add(&global_sum.ref_w(), local_sum.w);
				}
			});
			// 4 compute new pallete from sum
			parallel_for_each(acc_view, pallete.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				if(pallete.get_extent().contains(idx.global))
				{
					int_4 sum = pallete_sum_view(idx.global);
					if(sum.w != 0)
					{
						pallete(idx.global) = float_3(float(sum.x) / sum.w, float(sum.y) / sum.w, float(sum.z) / sum.w);
					}
				}
			});
		}
	}

	inline void apply_pallete_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3, array_view<const float_3, 2> pallete, bool ciede00_delta = false)
	{
		static const int tile_size = 32;
		static const int max_pallete_height = 16;
		static const int max_pallete_width = 16;
		int pallete_height = pallete.get_extent()[0];
		int pallete_width = pallete.get_extent()[1];
		assert(pallete_height <= max_pallete_height);
		assert(pallete_width <= max_pallete_width);
		int pallete_size = pallete_height * pallete_width;
		dest_channel1.discard_data();
		dest_channel2.discard_data();
		dest_channel3.discard_data();
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float_3 pallete_cache[max_pallete_height][max_pallete_width + 1];
			if(idx.local[0] < pallete_height && idx.local[1] < pallete_width)
			{
				pallete_cache[idx.local[0]][idx.local[1]] = pallete(idx.local);
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			concurrency::index<2> gidx = idx.global;
			if(dest_channel1.get_extent().contains(gidx))
			{
				float_3 src_value(src_channel1(gidx), src_channel2(gidx), src_channel3(gidx));
				float_3 best_pallete_value;
				float min_sqdiff = FLT_MAX;
				for(int i = 0; i < pallete_size; i++)
				{
					int y = i / pallete_width;
					int x = i % pallete_width;
					float sqdiff = ciede00_delta ? ciede2000_delta_e(src_value, pallete_cache[y][x]) : ciede1976_delta_e(src_value, pallete_cache[y][x]);
					if(sqdiff < min_sqdiff)
					{
						min_sqdiff = sqdiff;
						best_pallete_value = pallete_cache[y][x];
					}
				}
				dest_channel1(gidx) = best_pallete_value.x;
				dest_channel2(gidx) = best_pallete_value.y;
				dest_channel3(gidx) = best_pallete_value.z;
			}
		});
	}

	inline void apply_pallete_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel, array_view<const float_3, 2> pallete, bool ciede00_delta = false
		, float scale = 1.0f)
	{
		static const int tile_size = 32;
		static const int max_pallete_height = 16;
		static const int max_pallete_width = 16;
		int pallete_height = pallete.get_extent()[0];
		int pallete_width = pallete.get_extent()[1];
		assert(pallete_height <= max_pallete_height);
		assert(pallete_width <= max_pallete_width);
		int pallete_size = pallete_height * pallete_width;
		dest_channel.discard_data();
		parallel_for_each(acc_view, dest_channel.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static float_3 pallete_cache[max_pallete_height][max_pallete_width + 1];
			if(idx.local[0] < pallete_height && idx.local[1] < pallete_width)
			{
				pallete_cache[idx.local[0]][idx.local[1]] = pallete(idx.local);
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			concurrency::index<2> gidx = idx.global;
			if(dest_channel.get_extent().contains(gidx))
			{
				float_3 src_value(src_channel1(gidx), src_channel2(gidx), src_channel3(gidx));
				int best_pallete_index;
				float min_sqdiff = FLT_MAX;
				for(int i = 0; i < pallete_size; i++)
				{
					int y = i / pallete_width;
					int x = i % pallete_width;
					float sqdiff = ciede00_delta ? ciede2000_delta_e(src_value, pallete_cache[y][x]) : ciede1976_delta_e(src_value, pallete_cache[y][x]);
					if(sqdiff < min_sqdiff)
					{
						min_sqdiff = sqdiff;
						best_pallete_index = y * pallete_width + x;
					}
				}
				dest_channel(gidx) = float(best_pallete_index) * scale;
			}
		});
	}
}
