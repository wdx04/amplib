#pragma once

#include "amp_core.h"
#include "amp_warp_affine.h"

namespace amp
{
	inline void load_image_rois_32f_c1(vision_context& ctx, const cv::Mat& src_mat, const std::vector<cv::Rect> rois, array_view<float, 2> dest_array)
	{
		concurrency::array<float, 2> gpu_src_mat(src_mat.rows, src_mat.cols, ctx.acc_view);
		concurrency::array<int_4, 1> rois_mat(int(rois.size()), ctx.acc_view);
		ctx.load_cv_mat(src_mat, gpu_src_mat);
		concurrency::copy_async((const int_4*)&rois[0], rois_mat);
		concurrency::extent<2> ext(rois[0].height, rois[0].width);
		int roi_count = int(rois.size());
		int cols = ext[1];
		concurrency::parallel_for_each(ctx.acc_view, ext, [=, &gpu_src_mat, &rois_mat](concurrency::index<2> idx) restrict(amp) {
			for (int i = 0; i < roi_count; i++)
			{
				int_4 roi = rois_mat(i);
				int dest_index = idx[0] * cols + idx[1];
				concurrency::index<2> src_index(roi.y + idx[0], roi.x + idx[1]);
				dest_array(i, dest_index) = gpu_src_mat(src_index);
			}
		});
	}

	inline void load_images_32f_c1(vision_context& ctx, const std::vector<cv::Mat>& src_mats, concurrency::array<float, 2>& dest_array)
	{
		if (!src_mats.empty())
		{
			concurrency::array_view<float, 3> dest_array_view = dest_array.view_as<3>(concurrency::extent<3>(dest_array.extent[0], src_mats[0].rows, src_mats[0].cols));
			for (int i = 0; i < int(src_mats.size()); i++)
			{
				ctx.load_cv_mat(src_mats[i], dest_array_view[i]);
			}
		}
	}

	inline kernel_wrapper<float, 6U> get_euclidean_transform(float translation_x, float translation_y, float rotation)
	{
		kernel_wrapper<float, 6U> result;
		float radian = float(rotation * CV_PI / 180.0);
		result.data[0] = std::cosf(radian);
		result.data[1] = -std::sinf(radian);
		result.data[2] = translation_x;
		result.data[3] = std::sinf(radian);
		result.data[4] = std::cosf(radian);
		result.data[5] = translation_y;
		return result;
	}

	inline void load_ref_images(vision_context& ctx, const std::vector<cv::Mat>& ref_images, int translation_steps = 2, int rotation_steps = 10
		, float border_value = 255.0F, float translation_interval = 1.0f, float rotation_interval = 0.1f)
	{
		// ref images are loaded into ctx.float2d[0],[1],...
		int translation_count = translation_steps * 2 + 1;
		int rotation_count = rotation_steps * 2 + 1;
		int morph_count_per_image = translation_count * translation_count * rotation_count;
		for (int i = 0; i < int(ref_images.size()); i++)
		{
			concurrency::array<float, 2> gpu_ref_image(ref_images[i].rows, ref_images[i].cols);
			concurrency::array_view<float, 3> gpu_result_image = ctx.float2d[i].view_as<3>(concurrency::extent<3>(morph_count_per_image, ref_images[i].rows, ref_images[i].cols));
			ctx.load_cv_mat(ref_images[i], gpu_ref_image);
			for (int jx = -translation_steps; jx <= translation_steps; jx++)
			{
				for (int jy = -translation_steps; jy <= translation_steps; jy++)
				{
					for (int k = -rotation_steps; k <= rotation_steps; k++)
					{
						int index = (jx + translation_steps) * rotation_count * translation_count + (jy + translation_steps) * rotation_count + (k + rotation_steps);
						float translation_x = float(jx) * translation_interval;
						float translation_y = float(jy) * translation_interval;
						float rotation = float(k) * rotation_interval;
						amp::warp_affine_linear_32f_c1(ctx.acc_view, gpu_ref_image, gpu_result_image[index], get_euclidean_transform(translation_x, translation_y, rotation), border_value);
					}
				}
			}
			gpu_result_image.synchronize();
		}
	}

	inline void naive_batch_diff_32f_c1(accelerator_view& acc_view, array_view<const float, 2> input_array, array_view<const float, 2> ref_array, array_view<float, 2> diff_array)
	{
		static const int tile_size_input = 4;
		static const int tile_size_ref = 8;
		int input_count = input_array.extent[0];
		int ref_count = ref_array.extent[0];
		int cols = input_array.extent[1];
		concurrency::parallel_for_each(acc_view, diff_array.extent.tile<tile_size_input, tile_size_ref>().pad(), [=](concurrency::tiled_index<tile_size_input, tile_size_ref> idx) restrict(amp) {
			int input_index = idx.global[0];
			int ref_index = idx.global[1];
			if (input_index < input_count && ref_index < ref_count)
			{
				float sum = 0.0;
				for (int i = 0; i < cols; i++)
				{
					sum += fast_math::powf(fast_math::fabsf(input_array(input_index, i) - ref_array(ref_index, i)), 2.0f);
				}
				diff_array(idx.global) = fast_math::sqrtf(sum);
			}
		});
	}

	inline void batch_diff_32f_c1(accelerator_view& acc_view, array_view<const float, 1> input_array, array_view<const float, 2> ref_array, array_view<float, 1> diff_array)
	{
		static const int cache_size = 32;
		static const int block_multiplier = 8;
		static const int block_size = cache_size * block_multiplier;
		static const int tile_size = cache_size * cache_size;
		int ref_count = ref_array.extent[0];
		int cols = ref_array.extent[1];
		concurrency::extent<1> ext(ref_count * cache_size);
		concurrency::parallel_for_each(acc_view, ext.tile<tile_size>().pad(), [=](concurrency::tiled_index<tile_size> idx) restrict(amp) {
			tile_static float input_cache[block_size];
			tile_static float output_cache[cache_size][cache_size + 1];
			int ref_index = idx.global[0] / cache_size;
			int local_row = idx.local[0] / cache_size;
			int local_col = idx.local[0] % cache_size;
			float sum = 0.0f;
			int i = 0;
			for (; i < ROUNDDOWN(cols, block_size); i += block_size)
			{
				int col_index = i + local_col;
				if (local_row < block_multiplier)
				{
					input_cache[local_col + local_row * cache_size] = amp::guarded_read(input_array, concurrency::index<1>(col_index + local_row * cache_size));
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::powf(input_cache[local_col] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 2] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 2)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 3] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 3)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 4] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 4)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 5] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 5)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 6] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 6)), 2.0f);
				sum += fast_math::powf(input_cache[local_col + cache_size * 7] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 7)), 2.0f);
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			for (; i < ROUNDUP(cols, cache_size); i += cache_size)
			{
				if (local_row == 0)
				{
					input_cache[local_col] = amp::guarded_read(input_array, concurrency::index<1>(i + local_col));
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::powf(input_cache[local_col] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, i + local_col)), 2.0f);
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			output_cache[local_row][local_col] = sum;
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 2) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 2];
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 4) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 4];
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 8) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 8];
			idx.barrier.wait_with_tile_static_memory_fence();
			if(local_col == 0) amp::guarded_write(diff_array, concurrency::index<1>(ref_index), fast_math::sqrtf(output_cache[local_row][0] + output_cache[local_row][1] + output_cache[local_row][2] + output_cache[local_row][3]));
		});
	}

	inline void batch_diff_32f_c1(accelerator_view& acc_view, array_view<const float, 2> input_array, array_view<const float, 2> ref_array, array_view<float, 2> diff_array)
	{
		static const int tile_size = 14;
		int cols = input_array.extent[1];
		concurrency::parallel_for_each(acc_view, diff_array.extent.tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp) {
			tile_static float input_cache[tile_size][tile_size];
			tile_static float ref_cache[tile_size][tile_size];
			int input_start_index = idx.tile_origin[0];
			int ref_start_index = idx.tile_origin[1];
			int local_input_index = idx.local[0];
			int local_ref_index = idx.local[1];
			float sum = 0.0;
			for (int i = 0; i < ROUNDUP(cols, tile_size); i += tile_size)
			{
				input_cache[local_input_index][local_ref_index] = amp::guarded_read(input_array, concurrency::index<2>(input_start_index + local_input_index, i + local_ref_index));
				ref_cache[local_input_index][local_ref_index] = amp::guarded_read(ref_array, concurrency::index<2>(ref_start_index + local_input_index, i + local_ref_index));
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::powf(input_cache[local_input_index][0] - ref_cache[local_ref_index][0], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][1] - ref_cache[local_ref_index][1], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][2] - ref_cache[local_ref_index][2], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][3] - ref_cache[local_ref_index][3], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][4] - ref_cache[local_ref_index][4], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][5] - ref_cache[local_ref_index][5], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][6] - ref_cache[local_ref_index][6], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][7] - ref_cache[local_ref_index][7], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][8] - ref_cache[local_ref_index][8], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][9] - ref_cache[local_ref_index][9], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][10] - ref_cache[local_ref_index][10], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][11] - ref_cache[local_ref_index][11], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][12] - ref_cache[local_ref_index][12], 2.0f);
				sum += fast_math::powf(input_cache[local_input_index][13] - ref_cache[local_ref_index][13], 2.0f);
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			amp::guarded_write(diff_array, idx.global, fast_math::sqrtf(sum));
		});
	}

	inline void batch_diff_with_thresh_32f_c1(accelerator_view& acc_view, array_view<const float, 2> input_array, array_view<const float, 2> ref_array, array_view<float, 2> diff_array, float thresh)
	{
		static const int tile_size = 14;
		int cols = input_array.extent[1];
		concurrency::parallel_for_each(acc_view, diff_array.extent.tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp) {
			tile_static float input_cache[tile_size][tile_size];
			tile_static float ref_cache[tile_size][tile_size];
			int input_start_index = idx.tile_origin[0];
			int ref_start_index = idx.tile_origin[1];
			int local_input_index = idx.local[0];
			int local_ref_index = idx.local[1];
			int sum = 0;
			for (int i = 0; i < ROUNDUP(cols, tile_size); i += tile_size)
			{
				input_cache[local_input_index][local_ref_index] = amp::guarded_read(input_array, concurrency::index<2>(input_start_index + local_input_index, i + local_ref_index));
				ref_cache[local_input_index][local_ref_index] = amp::guarded_read(ref_array, concurrency::index<2>(ref_start_index + local_input_index, i + local_ref_index));
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::fabsf(input_cache[local_input_index][0] - ref_cache[local_ref_index][0]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][1] - ref_cache[local_ref_index][1]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][2] - ref_cache[local_ref_index][2]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][3] - ref_cache[local_ref_index][3]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][4] - ref_cache[local_ref_index][4]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][5] - ref_cache[local_ref_index][5]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][6] - ref_cache[local_ref_index][6]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][7] - ref_cache[local_ref_index][7]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][8] - ref_cache[local_ref_index][8]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][9] - ref_cache[local_ref_index][9]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][10] - ref_cache[local_ref_index][10]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][11] - ref_cache[local_ref_index][11]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][12] - ref_cache[local_ref_index][12]) > thresh;
				sum += fast_math::fabsf(input_cache[local_input_index][13] - ref_cache[local_ref_index][13]) > thresh;
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			amp::guarded_write(diff_array, idx.global, float(sum));
		});
	}

	inline void batch_diff_with_thresh_32f_c1(accelerator_view& acc_view, array_view<const float, 1> input_array, array_view<const float, 2> ref_array, array_view<float, 1> diff_array, float thresh)
	{
		static const int cache_size = 32;
		static const int block_multiplier = 8;
		static const int block_size = cache_size * block_multiplier;
		static const int tile_size = cache_size * cache_size;
		int ref_count = ref_array.extent[0];
		int cols = ref_array.extent[1];
		concurrency::extent<1> ext(ref_count * cache_size);
		concurrency::parallel_for_each(acc_view, ext.tile<tile_size>().pad(), [=](concurrency::tiled_index<tile_size> idx) restrict(amp) {
			tile_static float input_cache[block_size];
			tile_static int output_cache[cache_size][cache_size + 1];
			int ref_index = idx.global[0] / cache_size;
			int local_row = idx.local[0] / cache_size;
			int local_col = idx.local[0] % cache_size;
			int sum = 0;
			int i = 0;
			for (; i < ROUNDDOWN(cols, block_size); i += block_size)
			{
				int col_index = i + local_col;
				if (local_row < block_multiplier)
				{
					input_cache[local_col + local_row * cache_size] = amp::guarded_read(input_array, concurrency::index<1>(col_index + local_row * cache_size));
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::fabsf(input_cache[local_col] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 2] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 2))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 3] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 3))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 4] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 4))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 5] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 5))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 6] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 6))) > thresh;
				sum += fast_math::fabsf(input_cache[local_col + cache_size * 7] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, col_index + cache_size * 7))) > thresh;
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			for (; i < ROUNDUP(cols, cache_size); i += cache_size)
			{
				if (local_row == 0)
				{
					input_cache[local_col] = amp::guarded_read(input_array, concurrency::index<1>(i + local_col));
				}
				idx.barrier.wait_with_tile_static_memory_fence();
				sum += fast_math::fabsf(input_cache[local_col] - amp::guarded_read(ref_array, concurrency::index<2>(ref_index, i + local_col))) > thresh;
				idx.barrier.wait_with_tile_static_memory_fence();
			}
			output_cache[local_row][local_col] = sum;
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 2) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 2];
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 4) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 4];
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col < cache_size / 8) output_cache[local_row][local_col] += output_cache[local_row][local_col + cache_size / 8];
			idx.barrier.wait_with_tile_static_memory_fence();
			if (local_col == 0) amp::guarded_write(diff_array, concurrency::index<1>(ref_index), float(output_cache[local_row][0] + output_cache[local_row][1] + output_cache[local_row][2] + output_cache[local_row][3]));
		});
	}
}
