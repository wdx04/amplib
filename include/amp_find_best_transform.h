#pragma once

#include "amp_core.h"

namespace amp
{
	// find the best affine transform from template image to target image
	// brute-force search within a series of candidate transforms
	inline std::pair<int, cv::Matx23f> find_best_transform(concurrency::accelerator_view& acc_view, concurrency::array_view<const float, 2> templ_array,
		concurrency::array_view<const float, 2> target_array, const std::vector<cv::Matx23f>& transforms)
	{
		using namespace concurrency;
		concurrency::array<float, 2> gpu_transforms(concurrency::extent<2>(transforms.size(), 6), acc_view);
		concurrency::copy(reinterpret_cast<const float*>(&transforms[0]), gpu_transforms);
		concurrency::array_view<const float, 2> transform_array = gpu_transforms;
		std::vector<int> cpu_diffs(transforms.size());
		concurrency::array_view<int, 1> gpu_diffs(int(cpu_diffs.size()), &cpu_diffs[0]);
		int src_rows = templ_array.get_extent()[0];
		int src_cols = templ_array.get_extent()[1];
		int dst_rows = target_array.get_extent()[0];
		int dst_cols = target_array.get_extent()[1];
		int tr_count = gpu_transforms.get_extent()[0];
		static const int tile_size = 16;
		parallel_for_each(acc_view, target_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int dx = idx.global[1];
				int dy = idx.global[0];

				float dx_f = float(dx);
				float dy_f = float(dy);

				if (dx < dst_cols && dy < dst_rows)
				{
					float dest_val = target_array(idx.global);

					for (int trid = 0; trid < tr_count; trid++)
					{
						float sx_f = direct3d::mad(transform_array(trid, 0), dx_f, direct3d::mad(transform_array(trid, 1), dy_f, transform_array(trid, 2)));
						float sy_f = direct3d::mad(transform_array(trid, 3), dx_f, direct3d::mad(transform_array(trid, 4), dy_f, transform_array(trid, 5)));

						int sx = int(fast_math::floorf(sx_f));
						int sy = int(fast_math::floorf(sy_f));

						float ax = sx_f - float(sx);
						float ay = sy_f - float(sy);

						bool valid_src = sx >= 0 && sx + 1 < src_cols && sy >= 0 && sy + 1 < src_rows;
						if (valid_src)
						{
							float v0 = templ_array(sy, sx);
							float v1 = templ_array(sy, sx + 1);
							float v2 = templ_array(sy + 1, sx);
							float v3 = templ_array(sy + 1, sx + 1);
							float ax2 = 1.0f - ax, ay2 = 1.0f - ay;
							float val = direct3d::mad(ax2, direct3d::mad(v0, ay2, v2 * ay), ax * direct3d::mad(v1, ay2, v3 * ay));
							int diff = int(fast_math::fabsf(val - dest_val));
							concurrency::atomic_fetch_add(&gpu_diffs[trid], diff);
						}
					}
				}
			});
		gpu_diffs.synchronize();
		auto min_diff_index = std::min_element(cpu_diffs.cbegin(), cpu_diffs.cend()) - cpu_diffs.cbegin();
		return { cpu_diffs[min_diff_index], transforms[min_diff_index] };
	}
}
