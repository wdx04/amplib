#pragma once

// SLIC Superpixels
#include "amp_core.h"
#include "amp_filter2d.h"
#include "amp_ccl.h"
#include "amp_moments.h"
#include "amp_lut.h"

namespace amp
{
	struct superpixel
	{
	public:
		float y, x, l, a, b;

		superpixel()  restrict(cpu, amp)
			: y(0.0f), x(0.0f), l(0.0f), a(0.0f), b(0.0f)
		{
		}

		superpixel(float y_, float x_, float l_, float a_, float b_) restrict(cpu, amp)
			: y(y_), x(x_), l(l_), a(a_), b(b_)
		{
		}
	};

	struct superpixel_sum
	{
	public:
		int y, x, l, a, b, s;

		superpixel_sum() restrict(cpu, amp)
			: y(0), x(0), l(0), a(0), b(0), s(0)
		{
		}

		superpixel_sum(int y_, int x_, int l_, int a_, int b_, int s_) restrict(cpu, amp)
			: y(y_), x(x_), l(l_), a(a_), b(b_), s(s_)
		{
		}
	};

	inline void initialize_superpixels_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 3> superpixels)
	{
		int subpix_height = superpixels.get_extent()[0];
		int subpix_width = superpixels.get_extent()[1];
		float subpix_step_h = float(src_channel1.get_extent()[0]) / float(subpix_height);
		float subpix_step_w = float(src_channel1.get_extent()[1]) / float(subpix_width);
		concurrency::extent<2> ext(subpix_height, subpix_width);
		parallel_for_each(acc_view, ext, [=](concurrency::index<2> idx) restrict(amp) {
			// TODO choose a pixel with lowest gradient in 3x3 region
			float y = (float(idx[0]) + 0.5f) * subpix_step_h;
			float x = (float(idx[1]) + 0.5f) * subpix_step_w;
			concurrency::index<2> loc(int(fast_math::roundf(y)), int(fast_math::roundf(x)));
			superpixels(idx[0], idx[1], 0) = y;
			superpixels(idx[0], idx[1], 1) = x;
			superpixels(idx[0], idx[1], 2) = src_channel1(loc);
			superpixels(idx[0], idx[1], 3) = src_channel2(loc);
			superpixels(idx[0], idx[1], 4) = src_channel3(loc);
		});
	}

	template<int array_height, int array_width>
	inline void local_sum_up(superpixel_sum(&local_sums)[array_height][array_width], int row, int col, int y, int x, int l, int a, int b) restrict(amp)
	{
		superpixel_sum& local_sum = local_sums[row][col];
		concurrency::atomic_fetch_add(&local_sum.y, y);
		concurrency::atomic_fetch_add(&local_sum.x, x);
		concurrency::atomic_fetch_add(&local_sum.l, l);
		concurrency::atomic_fetch_add(&local_sum.a, a);
		concurrency::atomic_fetch_add(&local_sum.b, b);
		concurrency::atomic_fetch_inc(&local_sum.s);
	}

	inline void generate_superpixels_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 3> superpixels, float m = 1.0f, int iterations = 1)
	{
		static const int tile_size = 32;
		static const int neighbour_size = 7;
		static const int neighbour_count = neighbour_size * neighbour_size;
		int subpix_height = superpixels.get_extent()[0];
		int subpix_width = superpixels.get_extent()[1];
		float subpix_step_h = float(src_channel1.get_extent()[0]) / float(subpix_height);
		float subpix_step_w = float(src_channel1.get_extent()[1]) / float(subpix_width);
		float sqsinterval = float(src_channel1.get_extent().size()) / float(subpix_height * subpix_width);
		float sqm = m * m;
		concurrency::array<int, 3> global_sum(concurrency::extent<3>(superpixels.get_extent()[0], superpixels.get_extent()[1], superpixels.get_extent()[2] + 1), acc_view);
		for (int it = 0; it < iterations; it++)
		{
			int neighbour_dist = 0;
			if (it < iterations / 8)
			{
				neighbour_dist = 2;
			}
			else if (it < iterations / 2)
			{
				neighbour_dist = 1;
			}
			// initialize global sum
			parallel_for_each(acc_view, global_sum.get_extent(), [&global_sum](concurrency::index<3> idx) restrict(amp)
			{
				global_sum(idx) = 0;
			});
			// iterate
			parallel_for_each(acc_view, src_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=, &global_sum](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp) {
				tile_static superpixel superpixel_cache[neighbour_size][neighbour_size];
				tile_static superpixel_sum local_sums[neighbour_size][neighbour_size];
				tile_static int superpixel_y0, superpixel_x0;

				int local_row = idx.local[0];
				int local_col = idx.local[1];
				int local_id = local_row * tile_size + local_col;

				// calculate superpixel cache offset
				if (local_id == 0)
				{
					int tile_center_y = idx.global[0] + tile_size / 2;
					int tile_center_x = idx.global[1] + tile_size / 2;
					int center_superpixel_y = int(fast_math::roundf(float(tile_center_y) / subpix_step_h - 0.5f));
					int center_superpixel_x = int(fast_math::roundf(float(tile_center_x) / subpix_step_w - 0.5f));
					superpixel_y0 = center_superpixel_y - neighbour_size / 2;
					superpixel_x0 = center_superpixel_x - neighbour_size / 2;
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				// initialize local sums and superpixel cache
				if (local_row < neighbour_size && local_col < neighbour_size)
				{
					local_sums[local_row][local_col] = superpixel_sum();
					superpixel& c = superpixel_cache[local_row][local_col];
					c.y = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 0), FLT_MAX);
					c.x = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 1), FLT_MAX);
					c.l = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 2), FLT_MAX);
					c.a = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 3), FLT_MAX);
					c.b = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 4), FLT_MAX);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				// process pixel
				concurrency::index<2> gidx = idx.global;
				if (src_channel1.get_extent().contains(gidx))
				{
					superpixel w(float(gidx[0]), float(gidx[1]), src_channel1(gidx), src_channel2(gidx), src_channel3(gidx));
					float min_sqdiff = FLT_MAX;
					int min_sqdiff_row = -1;
					int min_sqdiff_col = -1;
					for (int j = 0; j < neighbour_count; j++)
					{
						int superpixel_row = j / neighbour_size;
						int superpixel_col = j % neighbour_size;
						const superpixel& v = superpixel_cache[superpixel_row][superpixel_col];
						float diff_y = v.y - w.y;
						float diff_x = v.x - w.x;
						float diff_l = v.l - w.l;
						float diff_a = v.a - w.a;
						float diff_b = v.b - w.b;
						float color_sqdiff = direct3d::mad(diff_l, diff_l, direct3d::mad(diff_a, diff_a, diff_b * diff_b));
						float pos_sqdiff = direct3d::mad(diff_y, diff_y, diff_x * diff_x);
						float sqdiff = color_sqdiff + pos_sqdiff / sqsinterval * sqm;
						if (sqdiff < min_sqdiff)
						{
							min_sqdiff = sqdiff;
							min_sqdiff_row = superpixel_row;
							min_sqdiff_col = superpixel_col;
						}
					}
					// update best superpixel's sum
					int y = gidx[0];
					int x = gidx[1];
					int l = int(fast_math::roundf(w.l));
					int a = int(fast_math::roundf(w.a));
					int b = int(fast_math::roundf(w.b));
					local_sum_up(local_sums, min_sqdiff_row, min_sqdiff_col, y, x, l, a, b);
					// update 1-neighbour superpixel's sum
					if (neighbour_dist >= 1)
					{
						local_sum_up(local_sums, min_sqdiff_row - 1, min_sqdiff_col, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row, min_sqdiff_col - 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row, min_sqdiff_col + 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row + 1, min_sqdiff_col, y, x, l, a, b);
					}
					// update 2-neighbour superpixel's sum
					if (neighbour_dist >= 2)
					{
						local_sum_up(local_sums, min_sqdiff_row - 2, min_sqdiff_col, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row - 1, min_sqdiff_col - 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row - 1, min_sqdiff_col + 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row, min_sqdiff_col - 2, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row, min_sqdiff_col + 2, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row + 1, min_sqdiff_col - 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row + 1, min_sqdiff_col + 1, y, x, l, a, b);
						local_sum_up(local_sums, min_sqdiff_row + 2, min_sqdiff_col, y, x, l, a, b);
					}
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				bool is_valid_pos = local_row < neighbour_size && local_col < neighbour_size
					&& local_row + superpixel_y0 >= 0 && local_row + superpixel_y0 < global_sum.get_extent()[0]
					&& local_col + superpixel_x0 >= 0 && local_col + superpixel_x0 < global_sum.get_extent()[1];
				if (is_valid_pos)
				{
					superpixel_sum& local_sum = local_sums[local_row][local_col];
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 0), local_sum.y);
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 1), local_sum.x);
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 2), local_sum.l);
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 3), local_sum.a);
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 4), local_sum.b);
					concurrency::atomic_fetch_add(&global_sum(local_row + superpixel_y0, local_col + superpixel_x0, 5), local_sum.s);
				}
			});
			// compute new superpixel centers
			parallel_for_each(acc_view, concurrency::extent<2>(superpixels.get_extent()[0], superpixels.get_extent()[1]), [=, &global_sum](concurrency::index<2> idx) restrict(amp)
			{
				int y = global_sum(idx[0], idx[1], 0);
				int x = global_sum(idx[0], idx[1], 1);
				int l = global_sum(idx[0], idx[1], 2);
				int a = global_sum(idx[0], idx[1], 3);
				int b = global_sum(idx[0], idx[1], 4);
				int s = global_sum(idx[0], idx[1], 5);
				superpixels(idx[0], idx[1], 0) = float(y) / float(s);
				superpixels(idx[0], idx[1], 1) = float(x) / float(s);
				superpixels(idx[0], idx[1], 2) = float(l) / float(s);
				superpixels(idx[0], idx[1], 3) = float(a) / float(s);
				superpixels(idx[0], idx[1], 4) = float(b) / float(s);
			});
		}
	}

	inline void apply_superpixels_32f_c3(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_array, array_view<const float, 3> superpixels, float m = 1.0f)
	{
		static const int tile_size = 32;
		static const int neighbour_size = 7;
		static const int neighbour_count = neighbour_size * neighbour_size;
		int subpix_height = superpixels.get_extent()[0];
		int subpix_width = superpixels.get_extent()[1];
		float subpix_step_h = float(src_channel1.get_extent()[0]) / float(subpix_height);
		float subpix_step_w = float(src_channel1.get_extent()[1]) / float(subpix_width);
		float sqsinterval = float(src_channel1.get_extent().size()) / float(subpix_height * subpix_width);
		float sqm = m * m;
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			tile_static superpixel superpixel_cache[neighbour_size][neighbour_size];
			tile_static int superpixel_y0, superpixel_x0;

			int local_row = idx.local[0];
			int local_col = idx.local[1];
			int local_id = local_row * tile_size + local_col;

			// calculate superpixel cache offset
			if (local_id == 0)
			{
				int tile_center_y = idx.global[0] + tile_size / 2;
				int tile_center_x = idx.global[1] + tile_size / 2;
				int center_superpixel_y = int(fast_math::roundf(float(tile_center_y) / subpix_step_h - 0.5f));
				int center_superpixel_x = int(fast_math::roundf(float(tile_center_x) / subpix_step_w - 0.5f));
				superpixel_y0 = center_superpixel_y - neighbour_size / 2;
				superpixel_x0 = center_superpixel_x - neighbour_size / 2;
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			// initialize local sums and superpixel cache
			if (local_row < neighbour_size && local_col < neighbour_size)
			{
				superpixel& c = superpixel_cache[local_row][local_col];
				c.y = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 0), FLT_MAX);
				c.x = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 1), FLT_MAX);
				c.l = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 2), FLT_MAX);
				c.a = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 3), FLT_MAX);
				c.b = guarded_read(superpixels, concurrency::index<3>(superpixel_y0 + local_row, superpixel_x0 + local_col, 4), FLT_MAX);
			}
			idx.barrier.wait_with_tile_static_memory_fence();

			concurrency::index<2> gidx = idx.global;
			if (dest_array.get_extent().contains(gidx))
			{
				superpixel w(float(gidx[0]), float(gidx[1]), src_channel1(gidx), src_channel2(gidx), src_channel3(gidx));
				float min_sqdiff = FLT_MAX;
				int min_sqdiff_row = -1;
				int min_sqdiff_col = -1;
				for (int j = 0; j < neighbour_count; j++)
				{
					int superpixel_row = j / neighbour_size;
					int superpixel_col = j % neighbour_size;
					const superpixel& v = superpixel_cache[superpixel_row][superpixel_col];
					float diff_y = v.y - w.y;
					float diff_x = v.x - w.x;
					float diff_l = v.l - w.l;
					float diff_a = v.a - w.a;
					float diff_b = v.b - w.b;
					float color_sqdiff = direct3d::mad(diff_l, diff_l, direct3d::mad(diff_a, diff_a, diff_b * diff_b));
					float pos_sqdiff = direct3d::mad(diff_y, diff_y, diff_x * diff_x);
					float sqdiff = color_sqdiff + pos_sqdiff / sqsinterval * sqm;
					if (sqdiff < min_sqdiff)
					{
						min_sqdiff = sqdiff;
						min_sqdiff_row = superpixel_row;
						min_sqdiff_col = superpixel_col;
					}
				}
				dest_array(gidx) = float((superpixel_y0 + min_sqdiff_row) * subpix_width + superpixel_x0 + min_sqdiff_col);
			}
		});
	}

	inline void enforce_superpixels_connectivity(accelerator_view& acc_view, array_view<float, 2> superpixel_array
		, array_view<int, 2> mask_array, array_view<int, 2> label_array, unsigned int area_thresh)
	{
		// 1st pass, merge into nearst cluster by centroid
		compute_connectivity_mask(acc_view, superpixel_array, mask_array, -0.5f, 0.5f);
		label_components(acc_view, mask_array, label_array);
		int blob_count = reorder_labels<3072U>(acc_view, label_array);
		concurrency::array<amp::moments, 1> moments_array(blob_count, acc_view);
		concurrency::array<float, 1> gpu_lut(blob_count, acc_view);
		calc_moments(acc_view, label_array, moments_array);
		concurrency::array<float_3, 1> large_labels(blob_count, acc_view);
		concurrency::array<float_3, 1> small_labels(blob_count, acc_view);
		concurrency::array<int, 1> label_count(2, acc_view);
		concurrency::parallel_for_each(acc_view, gpu_lut.get_extent(), [&label_count, &gpu_lut](concurrency::index<1> idx) restrict(amp)
		{
			if (idx[0] < 2) label_count[idx[0]] = 0;
			gpu_lut(idx) = float(idx[0]);
		});
		concurrency::parallel_for_each(acc_view, moments_array.get_extent(), [=, &label_count, &moments_array, &large_labels, &small_labels](concurrency::index<1> idx) restrict(amp)
		{
			const amp::moments& m = moments_array(idx);
			if (m.m00 >= area_thresh)
			{
				large_labels[concurrency::atomic_fetch_inc(&label_count[0])] = float_3(float(idx[0]), float(m.m10) / float(m.m00), float(m.m01) / float(m.m00));
			}
			else
			{
				small_labels[concurrency::atomic_fetch_inc(&label_count[1])] = float_3(float(idx[0]), float(m.m10) / float(m.m00), float(m.m01) / float(m.m00));
			}
		});
		std::vector<int> cpu_label_count(2);
		concurrency::copy(label_count, cpu_label_count.begin());
		int large_label_count = cpu_label_count[0];
		int small_label_count = cpu_label_count[1];
		if(small_label_count > 0)
		{
			concurrency::parallel_for_each(acc_view, concurrency::extent<1>(small_label_count), [=, &gpu_lut, &small_labels, &large_labels](concurrency::index<1> idx) restrict(amp)
			{
				float min_dist = FLT_MAX;
				float min_dist_label = -1.0f;
				int i = idx[0];
				float x = small_labels[i].y;
				float y = small_labels[i].z;
				for(int j = 0; j < large_label_count; j++)
				{
					float dist_x = x - large_labels[j].y;
					float dist_y = y - large_labels[j].z;
					float dist = dist_x * dist_x + dist_y * dist_y;
					if(dist < min_dist)
					{
						min_dist = dist;
						min_dist_label = large_labels[j].x;
					}
				}
				gpu_lut[int(small_labels[i].x)] = min_dist_label;
			});
			amp::lut_32f_c1(acc_view, label_array, gpu_lut, superpixel_array);
			// 2nd pass, merge into a random large neighbour
			compute_connectivity_mask(acc_view, superpixel_array, mask_array, -0.5f, 0.5f);
			label_components(acc_view, mask_array, label_array);
			blob_count = reorder_labels(acc_view, label_array);
			calc_moments(acc_view, label_array, moments_array);
			concurrency::array<int, 1> label_flags(blob_count, acc_view);
			concurrency::array<int, 1> label_flags2(blob_count, acc_view);
			concurrency::parallel_for_each(acc_view, gpu_lut.get_extent(), [&gpu_lut](concurrency::index<1> idx) restrict(amp)
			{
				gpu_lut(idx) = float(idx[0]);
			});
			concurrency::parallel_for_each(acc_view, concurrency::extent<1>(blob_count), [=, &moments_array, &label_flags, &label_flags2](concurrency::index<1> idx) restrict(amp)
			{
				const amp::moments& m = moments_array(idx);
				int flag = m.m00 >= area_thresh ? 1 : 0;
				label_flags(idx) = flag;
				label_flags2(idx) = flag;
			});
			// find a random neighbour
			accelerator_view cpu_acc_view = concurrency::accelerator(concurrency::accelerator::cpu_accelerator).create_view();
			concurrency::array<int, 1> staging_label_changed(1, cpu_acc_view, acc_view);
			array_view<int, 1> label_changed(staging_label_changed);
			static const int tile_size = 32;
			static const int tile_static_size_3x3 = tile_size + 2;
			do
			{
				label_changed[0] = 0;
				concurrency::parallel_for_each(acc_view, label_array.get_extent().tile<tile_size, tile_size>().pad(), [label_array, label_changed, &label_flags, &gpu_lut](tiled_index<tile_size, tile_size> idx) restrict(amp) {
					tile_static int data[tile_size + 2][tile_size + 2];
					int row = idx.global[0];
					int col = idx.global[1];
					int local_row = idx.local[0];
					int local_col = idx.local[1];
					int dx = row - local_row - 1;
					int dy = col - local_col - 1;
					const int id = direct3d::imin(local_row * tile_size + local_col, tile_static_size_3x3 * tile_static_size_3x3 / 2 - 1);
					int dr = id / tile_static_size_3x3;
					int dc = id % tile_static_size_3x3;
					data[dr][dc] = guarded_read_reflect101(label_array, concurrency::index<2>(dx + dr, dy + dc));
					data[dr + tile_static_size_3x3 / 2][dc] = guarded_read_reflect101(label_array, concurrency::index<2>(dx + dr + tile_static_size_3x3 / 2, dy + dc));
					idx.barrier.wait_with_tile_static_memory_fence();

					int current_label = data[local_row + 1][local_col + 1];
					if(label_flags(current_label) == 0)
					{
						int up_label = data[local_row][local_col + 1];
						int down_label = data[local_row + 2][local_col + 1];
						int left_label = data[local_row + 1][local_col];
						int right_label = data[local_row + 1][local_col + 2];
						int neighbour_label = -1;
						neighbour_label = current_label != up_label && label_flags(up_label) != 0 ? up_label : neighbour_label;
						neighbour_label = current_label != down_label && label_flags(down_label) != 0 ? down_label : neighbour_label;
						neighbour_label = current_label != left_label && label_flags(left_label) != 0 ? left_label : neighbour_label;
						neighbour_label = current_label != right_label && label_flags(right_label) != 0 ? right_label : neighbour_label;
						if(neighbour_label != -1)
						{
							gpu_lut(current_label) = float(neighbour_label);
							label_flags(current_label) = 1;
							label_changed[0] = 1;
						}
					}
				});
			} while(label_changed[0] != 0);
			concurrency::parallel_for_each(acc_view, concurrency::extent<1>(blob_count), [=, &label_flags2, &gpu_lut](concurrency::index<1> idx) restrict(amp)
			{
				int label = idx[0];
				int mapped_label = int(gpu_lut(idx));
				if(label != mapped_label)
				{
					while(label_flags2(mapped_label) == 0)
					{
						mapped_label = int(gpu_lut(mapped_label));
					}
					gpu_lut(label) = float(mapped_label);
				}
			});
			amp::lut_32f_c1(acc_view, label_array, gpu_lut, superpixel_array);
		}
	}

	inline void draw_superpixels_boundary_32f_c3(accelerator_view& acc_view, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3, array_view<const float, 2> superpixel_array)
	{
		static const int tile_size = 32;
		parallel_for_each(acc_view, superpixel_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if (superpixel_array.get_extent().contains(idx.global))
			{
				int row = idx.global[0];
				int col = idx.global[1];
				float self = superpixel_array(idx.global);
				float up = guarded_read_reflect101(superpixel_array, concurrency::index<2>(row - 1, col));
				float down = guarded_read_reflect101(superpixel_array, concurrency::index<2>(row + 1, col));
				float left = guarded_read_reflect101(superpixel_array, concurrency::index<2>(row, col - 1));
				float right = guarded_read_reflect101(superpixel_array, concurrency::index<2>(row, col + 1));
				bool is_edge = self != up || self != down || self != left || self != right;
				if (is_edge)
				{
					dest_channel1(idx.global) = 255.0f;
					dest_channel2(idx.global) = 255.0f;
					dest_channel3(idx.global) = 255.0f;
				}
			}
		});
	}
}
