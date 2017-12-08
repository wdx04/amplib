#pragma once

#include "amp_core.h"
#include "amp_canny.h"
#include "amp_count_nonzero.h"

namespace amp
{
	inline int hough_circles_32f_c1(canny_context& ctx, array_view<const float, 2> src_array, array_view<float, 2> edges_array, array_view<float_4, 1> circles, float dp, float minDist
		, float cannyHighThresh, int votesThreshold, int minRadius, int maxRadius, bool skip_canny = false)
	{
		int maxCircles = circles.get_extent()[0];
		// 1.Canny edge detection or Reuse previous canny result
		if (!skip_canny)
		{
			canny_32f_c1(ctx, src_array, edges_array, cannyHighThresh / 2.0f, cannyHighThresh, true);
		}
		
		// 2.Build edge point list
		int pt_count = count_nonzero_32f_c1(ctx.acc_view, edges_array);
		if (pt_count == 0) return 0;
		static const int pt_pixels_per_wi = 16;
		static const int tile_size = 32;
		static const int tile_static_size = tile_size * pt_pixels_per_wi;
		concurrency::array<int_2, 1> pt_list(pt_count, ctx.acc_view);
		concurrency::array<int, 1> global_offset(3, ctx.acc_view);
		concurrency::extent<2> ext_pt_list(edges_array.get_extent()[0], DIVUP(edges_array.get_extent()[1], pt_pixels_per_wi));
		parallel_for_each(ctx.acc_view, concurrency::extent<1>(3), [&global_offset](concurrency::index<1> idx) restrict(amp)
		{
			global_offset(idx) = 0;
		});
		parallel_for_each(ctx.acc_view, ext_pt_list.tile<1, tile_size>().pad(), [=, &global_offset, &pt_list](tiled_index<1, tile_size> idx) restrict(amp)
		{
			tile_static int_2 local_pt_list[tile_static_size];
			tile_static int local_idx;
			tile_static int local_offset;
			int lid = idx.local[1];
			int x = idx.global[1] * pt_pixels_per_wi;
			int y = idx.global[0];
			if (lid == 0) local_idx = 0;
			idx.barrier.wait_with_all_memory_fence();

			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			if (guarded_read(edges_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = int_2(x, y); x++;
			idx.barrier.wait_with_tile_static_memory_fence();

			if (idx.local[1] == 0) local_offset = concurrency::atomic_fetch_add(&global_offset[0], local_idx);
			idx.barrier.wait_with_tile_static_memory_fence();

			for (int i = lid; i < local_idx; i += tile_size)
			{
				pt_list[local_offset + i] = local_pt_list[i];
			}
		});

		// 3.Vote for circle centers
		static const int tile_size_1d = 256;
		array_view<int, 2> dx(ctx.dx);
		array_view<int, 2> dy(ctx.dy);
		const float idp = 1.0f / dp;
		const int height = edges_array.get_extent()[0];
		const int width = edges_array.get_extent()[1];
		concurrency::array<int, 2> accum(cvCeil(edges_array.get_extent()[0] * idp) + 2, cvCeil(edges_array.get_extent()[1] * idp) + 2, ctx.acc_view);
		parallel_for_each(ctx.acc_view, pt_list.get_extent().tile<tile_size_1d>().pad(), [=, &accum, &pt_list](const tiled_index<tile_size_1d> idx) restrict(amp)
		{
			const int SHIFT = 10;
			const int ONE = 1 << SHIFT;

			const int tid = idx.global[0];

			if (tid >= pt_count)
				return;

			int_2 val = pt_list(tid);

			const int x = val.x;
			const int y = val.y;

			const int vx = dx(y, x);
			const int vy = dy(y, x);

			if (vx == 0 && vy == 0)
				return;

			const float mag = fast_math::sqrtf(float(vx * vx + vy * vy));

			const int x0 = int(fast_math::roundf((x * idp) * ONE));
			const int y0 = int(fast_math::roundf((y * idp) * ONE));

			int sx = int(fast_math::roundf((vx * idp) * ONE / mag));
			int sy = int(fast_math::roundf((vy * idp) * ONE / mag));

			// Step from minRadius to maxRadius in both directions of the gradient
			for (int k1 = 0; k1 < 2; ++k1)
			{
				int x1 = x0 + minRadius * sx;
				int y1 = y0 + minRadius * sy;

				for (int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, ++r)
				{
					const int x2 = x1 >> SHIFT;
					const int y2 = y1 >> SHIFT;

					if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
						break;

					concurrency::atomic_fetch_inc(&accum(y2 + 1, x2 + 1));
				}

				sx = -sx;
				sy = -sy;
			}
		});

		// 4.Build circle center list
		concurrency::array<int_2, 1> center_list(edges_array.get_extent().size(), ctx.acc_view);
		concurrency::parallel_for_each(ctx.acc_view, accum.get_extent().tile<tile_size, tile_size>().pad(), [=, &accum, &center_list, &global_offset](tiled_index<tile_size, tile_size> idx) restrict(amp) {
			const int x = idx.global[1];
			const int y = idx.global[0];

			if (x < accum.get_extent()[1] - 2 && y < accum.get_extent()[0] - 2)
			{
				const int top = accum(y, x + 1);
				const int left = accum(y + 1, x);
				const int cur = accum(y + 1, x + 1);
				const int right = accum(y + 1, x + 2);
				const int bottom = accum(y + 2, x + 1);
				if (cur > votesThreshold && cur > top && cur >= bottom && cur > left && cur >= right)
				{
					const int idx = concurrency::atomic_fetch_inc(&global_offset(1));
					center_list[idx] = int_2(x, y);
				}
			}
		});
		int center_count = 0;
		concurrency::copy(global_offset.section(1, 1), &center_count);
		if (center_count == 0) return 0;
		std::vector<std::pair<int, int>> cpu_centers(center_count);
		concurrency::copy(center_list.section(0, center_count), (int_2*)&cpu_centers[0]);
		cv::Mat cpu_accum(cvCeil(edges_array.get_extent()[0] * idp) + 2, cvCeil(edges_array.get_extent()[1] * idp) + 2, CV_32SC1);
		concurrency::copy(accum, cpu_accum.ptr<int>());

		// 5.Filter centers according to minDist on CPU
		if (minDist > 1)
		{
			std::vector<std::pair<int, int>> new_centers(center_count);
			int new_count = 0;
			const int cellSize = cvRound(minDist);
			const int gridWidth = (edges_array.get_extent()[1] + cellSize - 1) / cellSize;
			const int gridHeight = (edges_array.get_extent()[0] + cellSize - 1) / cellSize;

			std::vector<std::vector<std::pair<int, int>>> grid(gridWidth * gridHeight);

			const float minDist2 = minDist * minDist;

			std::vector<cv::Vec3f> sortBuf;
			for (int i = 0; i < center_count; i++){
				cv::Vec3f temp;
				temp[0] = float(cpu_centers[i].first);
				temp[1] = float(cpu_centers[i].second);
				temp[2] = float(cpu_accum.at<int>(cpu_centers[i].second + 1, cpu_centers[i].first + 1));
				sortBuf.push_back(temp);
			}
			std::sort(sortBuf.begin(), sortBuf.end(), [](cv::Vec3f a, cv::Vec3f b) {
				return a[2] > b[2];
			});

			for (int i = 0; i < center_count; ++i)
			{
				std::pair<int, int> p;
				p.first = cvRound(sortBuf[i][0]);
				p.second = cvRound(sortBuf[i][1]);

				bool good = true;

				int xCell = static_cast<int>(p.first / cellSize);
				int yCell = static_cast<int>(p.second / cellSize);

				int x1 = xCell - 1;
				int y1 = yCell - 1;
				int x2 = xCell + 1;
				int y2 = yCell + 1;

				// boundary check
				x1 = std::max(0, x1);
				y1 = std::max(0, y1);
				x2 = std::min(gridWidth - 1, x2);
				y2 = std::min(gridHeight - 1, y2);

				for (int yy = y1; yy <= y2; ++yy)
				{
					for (int xx = x1; xx <= x2; ++xx)
					{
						std::vector<std::pair<int,int>>& m = grid[yy * gridWidth + xx];

						for (size_t j = 0; j < m.size(); ++j)
						{
							float dx = (float)(p.first - m[j].first);
							float dy = (float)(p.second - m[j].second);

							if (dx * dx + dy * dy < minDist2)
							{
								good = false;
								goto break_out;
							}
						}
					}
				}

			break_out:
				if (good)
				{
					grid[yCell * gridWidth + xCell].push_back(p);
					new_centers[new_count++] = p;
				}
			}
			center_count = new_count;
			concurrency::copy_async((int_2*)&new_centers[0], (int_2*)(&new_centers[0] + center_count), center_list);
		}

		// 6.Find Radius
		static const int max_hist_size = 50;
		static const int radius_tile_size = 1024;
		concurrency::extent<1> radius_ext(center_count * radius_tile_size);
		const int hist_size = maxRadius - minRadius + 1;
		assert(hist_size + 2 <= max_hist_size);
		concurrency::parallel_for_each(ctx.acc_view, radius_ext.tile<radius_tile_size>().pad(), [=, &pt_list, &center_list, &global_offset](tiled_index<radius_tile_size> idx) restrict(amp) {
			tile_static int smem[max_hist_size];
			int lid = idx.local[0];
			for (int i = lid; i < hist_size + 2; i += idx.tile_dim0)
				smem[i] = 0;
			idx.barrier.wait_with_tile_static_memory_fence();

			int_2 val = center_list[idx.tile[0]];

			float cx = float(val.x);
			float cy = float(val.y);

			cx = (cx + 0.5f) * dp;
			cy = (cy + 0.5f) * dp;

			for (int i = lid; i < pt_count; i += idx.tile_dim0)
			{
				val = pt_list[i];

				const int x = val.x;
				const int y = val.y;

				const float rad = fast_math::sqrtf((cx - x) * (cx - x) + (cy - y) * (cy - y));
				if (rad >= minRadius && rad <= maxRadius)
				{
					const int r = int(fast_math::roundf(rad - minRadius));
					concurrency::atomic_fetch_inc(&smem[r + 1]);
				}
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			for (int i = lid; i < hist_size; i += idx.tile_dim0)
			{
				const int curVotes = smem[i + 1];

				if (curVotes >= votesThreshold && curVotes > smem[i] && curVotes >= smem[i + 2])
				{
					const int ind = concurrency::atomic_fetch_inc(&global_offset[2]);
					if (ind < maxCircles)
					{
						circles[ind] = float_4(cx, cy, float(i + minRadius), float(curVotes));
					}
				}
			}
		});

		int circles_count = 0;
		concurrency::copy(global_offset.section(2, 1), &circles_count);
		return std::min(circles_count, maxCircles);
	}
}
