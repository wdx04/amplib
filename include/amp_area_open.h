#pragma once

#include "amp_core.h"
#include <set>

namespace amp
{
	template<typename keep_fn_t>
	inline void binary_area_fill_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<int, 2> label_array, keep_fn_t keep_fn, float fill_value = 0.0f)
	{
		static const int tile_size = 32;
		// Copy labels to CPU
		std::vector<int> cpu_label_array(label_array.get_extent().size());
		concurrency::copy(label_array, &cpu_label_array[0]);
		// Sort labels
		std::sort(cpu_label_array.begin(), cpu_label_array.end());
		// Labels to be retained
		std::vector<int> cpu_labels_to_retain = keep_fn(cpu_label_array);
		// Upload to GPU
		concurrency::array<int, 1> labels_to_retain(int(cpu_labels_to_retain.size()), acc_view);
		concurrency::copy(&cpu_labels_to_retain[0], labels_to_retain);
		// Filter Image on GPU
		if (cpu_labels_to_retain.size() > 1)
		{
			static const int max_labels_to_retain = 1024;
			concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=, &labels_to_retain](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static int shared_labels_to_retain[max_labels_to_retain];
				int label_to_retain_count = direct3d::imin(labels_to_retain.get_extent()[0], max_labels_to_retain);
				int index = idx.local[0] * tile_size + idx.local[1];
				if (index < label_to_retain_count)
				{
					shared_labels_to_retain[index] = labels_to_retain(index);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				float src_value = guarded_read(src_array, idx.global);
				int label_value = guarded_read(label_array, idx.global);
				int start_index = 0;
				int end_index = label_to_retain_count;
				int mid_index = (start_index + end_index) / 2;
				bool label_found = false;
				do
				{
					int mid_value = shared_labels_to_retain[mid_index];
					if (mid_value == label_value)
					{
						label_found = true;
						break;
					}
					else if (label_value < mid_value)
					{
						end_index = mid_index;
					}
					else
					{
						start_index = mid_index + 1;
					}
					mid_index = (start_index + end_index) / 2;
				} while (start_index < end_index);
				guarded_write(dest_array, idx.global, label_found ? src_value : fill_value);
			});
		}
		else
		{
			kernel_wrapper<int, 1> label_to_remain(1, 1, { cpu_labels_to_retain[0] });
			concurrency::parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=, &labels_to_retain](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				float src_value = guarded_read(src_array, idx.global);
				int label_value = guarded_read(label_array, idx.global);
				guarded_write(dest_array, idx.global, label_value == label_to_remain.data[0] ? src_value : fill_value);
			});
		}
	}

	inline void binary_area_open_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<int, 2> label_array, int n)
	{
		binary_area_fill_32f_c1(acc_view, src_array, dest_array, label_array, [n](const std::vector<int>& cpu_label_array) -> std::vector<int> {
			std::vector<int> cpu_labels_to_retain;
			cpu_labels_to_retain.reserve(cpu_label_array.size());
			int current_label = cpu_label_array[0];
			int current_count = 0;
			for (size_t i = 0; i < cpu_label_array.size(); i++)
			{
				int label = cpu_label_array[i];
				if (label != current_label)
				{
					if (current_count >= n)
					{
						cpu_labels_to_retain.emplace_back(current_label);
					}
					current_label = label;
					current_count = 1;
				}
				else
				{
					current_count++;
				}
			}
			if (current_count >= n)
			{
				cpu_labels_to_retain.emplace_back(current_label);
			}
			return cpu_labels_to_retain;
		}, 0.0f);
	}

	// get largest blob from binary image
	// call binary_filter_labels before using this function
	inline void binary_k_largest_areas_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, array_view<int, 2> label_array, size_t k = 1)
	{
		binary_area_fill_32f_c1(acc_view, src_array, dest_array, label_array, [&src_array, k](const std::vector<int>& cpu_label_array) -> std::vector<int> {
			std::set<std::pair<int, int>, std::greater<std::pair<int,int>>> label_counts;
			int current_label = cpu_label_array[0];
			int current_count = 0;
			for (size_t i = 0; i < cpu_label_array.size(); i++)
			{
				int label = cpu_label_array[i];
				if (label < 0)
				{
					continue;
				}
				if (label != current_label)
				{
					label_counts.insert(std::make_pair(current_count, current_label));
					current_label = label;
					current_count = 1;
				}
				else
				{
					current_count++;
				}
			}
			label_counts.insert(std::make_pair(current_count, current_label));
			std::vector<int> cpu_label_to_retain;
			for (const auto& p : label_counts)
			{
				cpu_label_to_retain.push_back(p.second);
				if (cpu_label_to_retain.size() >= k) break;
			}
			return cpu_label_to_retain;
		}, 0.0f);
	}
}
