#pragma once

#include "amp_core.h"

namespace amp
{
	inline std::pair<float, float> mean_std_dev_32f_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray)
	{
		static const int tile_size = 1024;
		static const int tile_size_2a = 512;
#if OPTIMIZE_FOR_INTEL
		static const int group_count = 10;
#else
		static const int group_count = 64;
#endif
		concurrency::extent<1> ext(tile_size * group_count);
		concurrency::array<float, 1> part_sum(group_count, acc_view);
		concurrency::array<float, 1> part_sqsum(group_count, acc_view);
		concurrency::array_view<float, 1> part_sum_view(part_sum);
		concurrency::array_view<float, 1> part_sqsum_view(part_sqsum);
		int total = int(srcArray.get_extent().size());
		int cols = srcArray.get_extent()[1];
		part_sum_view.discard_data();
		part_sqsum_view.discard_data();
		parallel_for_each(acc_view, ext.tile<tile_size>(), [=](tiled_index<tile_size> idx) restrict(amp)
		{
			int lid = idx.local[0];
			int gid = idx.tile[0];
			int grain = group_count * tile_size;

			tile_static float localMemSum[tile_size_2a];
			tile_static float localMemSqSum[tile_size_2a];

			float accSum = 0.0f;
			float accSqSum = 0.0f;
			for(int id = idx.global[0]; id < total; id += grain)
			{
				float value = guarded_read(srcArray, concurrency::index<2>(id / cols, id % cols));
				accSum += value;
				accSqSum = direct3d::mad(value, value, accSqSum);
			}

			if(lid < tile_size_2a)
			{
				localMemSum[lid] = accSum;
				localMemSqSum[lid] = accSqSum;
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			if(lid >= tile_size_2a && total >= tile_size_2a)
			{
				localMemSum[lid - tile_size_2a] += accSum;
				localMemSqSum[lid - tile_size_2a] += accSqSum;
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			for(int lsize = tile_size_2a >> 1; lsize > 0; lsize >>= 1)
			{
				if(lid < lsize)
				{
					int lid2 = lsize + lid;
					localMemSum[lid] += localMemSum[lid2];
					localMemSqSum[lid] += localMemSqSum[lid2];
				}
				idx.barrier.wait_with_tile_static_memory_fence();
			}

			if(lid == 0)
			{
				part_sum_view(gid) = localMemSum[0];
				part_sqsum_view(gid) = localMemSqSum[0];
			}
		});
		std::vector<float> cpu_part_sum(group_count);
		std::vector<float> cpu_part_sqsum(group_count);
		copy(part_sum_view, cpu_part_sum.begin());
		copy(part_sqsum_view, cpu_part_sqsum.begin());
		float mean = std::accumulate(cpu_part_sum.begin(), cpu_part_sum.end(), 0.0f) / float(total);
		float stddev = std::sqrtf(std::max<float>(std::accumulate(cpu_part_sqsum.begin(), cpu_part_sqsum.end(), 0.0f) / float(total) - mean * mean, 0.0f));
		return std::pair<float, float>(mean, stddev);
	}

}
