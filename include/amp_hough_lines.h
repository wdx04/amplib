#pragma once

#include "amp_core.h"
#include "amp_count_nonzero.h"

namespace amp
{
	// Hough Lines
	// Note on min_theta and max_theta params
	// To detect horizontal lines within +/- delta range: min_theta = CV_PI / 2.0 - delta; max_theta = CV_PI / 2.0 + delta
	// TO detect vertical lines within +/- delta range: min_theta = CV_PI - delta; max_theta = delta
	inline int hough_lines_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float_3, 1> lines, float rho, float theta, int threshold, float min_theta, float max_theta)
	{
		// check parameters
		assert(max_theta >= 0.0f && max_theta <= float(CV_PI));
		assert(min_theta >= 0.0f && min_theta <= float(CV_PI));
		assert(rho > 0.0f && theta > 0.0f);

		// make point list
		int pt_count = count_nonzero_32f_c1(acc_view, src_array);
		if(pt_count == 0) return 0;
		static const int pt_pixels_per_wi = 16;
		static const int tile_size = 32;
		static const int tile_static_size = tile_size * pt_pixels_per_wi;
		concurrency::array<float_2, 1> pt_list(pt_count, acc_view);
		concurrency::array<int, 1> global_offset(2, acc_view);
		concurrency::extent<2> ext_pt_list(src_array.get_extent()[0], DIVUP(src_array.get_extent()[1], pt_pixels_per_wi));
		parallel_for_each(acc_view, concurrency::extent<1>(2), [&global_offset](concurrency::index<1> idx) restrict(amp)
		{
			global_offset(idx) = 0;
		});
		parallel_for_each(acc_view, ext_pt_list.tile<1, tile_size>().pad(), [=, &global_offset, &pt_list](tiled_index<1, tile_size> idx) restrict(amp)
		{
			tile_static float_2 local_pt_list[tile_static_size];
			tile_static int local_idx;
			tile_static int local_offset;
			int lid = idx.local[1];
			int x = idx.global[1] * pt_pixels_per_wi;
			int y = idx.global[0];
			if(lid == 0) local_idx = 0;
			idx.barrier.wait_with_all_memory_fence();

			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			if(guarded_read(src_array, concurrency::index<2>(y, x)) != 0.0f) local_pt_list[concurrency::atomic_fetch_inc(&local_idx)] = float_2(float(x), float(y)); x++;
			idx.barrier.wait_with_tile_static_memory_fence();

			if(idx.local[1] == 0) local_offset = concurrency::atomic_fetch_add(&global_offset[0], local_idx);
			idx.barrier.wait_with_tile_static_memory_fence();

			for(int i = lid; i < local_idx; i += tile_size)
			{
				pt_list[local_offset + i] = local_pt_list[i];
			}
		});

		// accumulate(use global memory)
		int numangle = max_theta > min_theta ? cvRound((max_theta - min_theta) / theta) : cvRound((max_theta + CV_PI - min_theta) / theta);
		int numrho = cvRound(((src_array.get_extent()[0] + src_array.get_extent()[1]) * 2 + 1) / rho);
		concurrency::array<int, 2> accum(numangle + 2, numrho + 2, acc_view);
		float irho = (float)(1 / rho);
		size_t local_memory_needed = (numrho + 2)*sizeof(int);

		parallel_for_each(acc_view, accum.get_extent(), [&accum](concurrency::index<2> idx) restrict(amp)
		{
			accum(idx) = 0;
		});
		int acc_wg_size = std::min(pt_count, 1024);
		float cv_pi_f = float(CV_PI);
		parallel_for_each(acc_view, concurrency::extent<2>(numangle, acc_wg_size), [=, &accum, &pt_list](concurrency::index<2> idx) restrict(amp)
		{
			int theta_idx = idx[0];
			int count_idx = idx[1];
			float sinVal;
			float cosVal;
			fast_math::sincosf(fast_math::fmodf(min_theta + theta * ((float)theta_idx), cv_pi_f), &sinVal, &cosVal);
			sinVal *= irho;
			cosVal *= irho;
			const int shift = (numrho - 1) / 2;
			for(int i = count_idx; i < pt_count; i += acc_wg_size)
			{
				const float_2 val = pt_list(i);
				int r = int(fast_math::roundf(direct3d::mad(val.x, cosVal, val.y * sinVal))) + shift;
				concurrency::atomic_fetch_inc(&accum[theta_idx + 1][r + 1]);
			}
		});

		// get lines
		static const int getline_pixels_per_wi = 8;
		int max_lines = lines.get_extent()[0];
		concurrency::extent<2> getline_ext(numangle, (numrho + getline_pixels_per_wi - 1) / getline_pixels_per_wi);
		parallel_for_each(acc_view, getline_ext, [=, &accum, &global_offset](concurrency::index<2> idx) restrict(amp)
		{
			int x0 = idx[1];
			int y = idx[0];
			int glob_size = getline_ext[1];

			for(int x = x0; x < numrho; x += glob_size)
			{
				int curVote = accum(y + 1, x + 1);
				if(curVote > threshold && curVote > accum(y + 1, x) && curVote >= accum(y + 1, x + 2) && curVote > accum(y, x + 1) && curVote >= accum(y + 2, x + 1))
				{
					int index = concurrency::atomic_fetch_inc(&global_offset[1]);
					if(index < max_lines)
					{
						float radius = (x - (numrho - 1) * 0.5f) * rho;
						float angle = fast_math::fmodf(min_theta + y * theta, cv_pi_f);
						lines[index] = float_3(radius, angle, float(curVote));
					}
				}
			}
		});
		int lines_count = 0;
		concurrency::copy(global_offset.section(1, 1), &lines_count);
		return std::min(lines_count, max_lines);
	}
}
