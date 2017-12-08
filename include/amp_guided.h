#pragma once

#include "amp_core.h"
#include "amp_box.h"

namespace amp
{
	// Guided Filter
	class guided_filter_context
	{
	public:
		guided_filter_context(const accelerator_view& acc_view_, int rows_, int cols_)
			: acc_view(acc_view_), mean_I(rows_, cols_, acc_view_), mean_p(rows_, cols_, acc_view_), mean_Ip(rows_, cols_, acc_view_)
			, mean_II(rows_, cols_, acc_view_), a(rows_, cols_, acc_view_), b(rows_, cols_, acc_view_)
		{
		}

		concurrency::array<float, 2> mean_I, mean_p, mean_Ip, mean_II, a, b;
		accelerator_view acc_view;
	};

	namespace detail
	{
		inline void box_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array1, array_view<const float, 2> src_array2, array_view<float, 2> dest_array, array_view<float, 2> row_temp_array, int ksize)
		{
			static const int tile_size = 32;
#if !OPTIMIZE_FOR_AMD
			if(ksize <= 215)
			{
				row_temp_array.discard_data();
				parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
				{
					guarded_write(row_temp_array, idx.global, guarded_read(src_array1, idx.global) * guarded_read(src_array2, idx.global));
				});
				box_filter_32f_c1(acc_view, row_temp_array, dest_array, ksize, ksize);
				return;
			}
#endif
			dest_array.discard_data();
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				guarded_write(dest_array, idx.global, guarded_read(src_array1, idx.global) * guarded_read(src_array2, idx.global));
			});
			::amp::box_filter_32f_c1(acc_view, dest_array, dest_array, row_temp_array, ksize, ksize);
		}
	}

	inline void guided_filter_32f_c1(guided_filter_context& ctx, array_view<const float, 2> guide_array, array_view<const float, 2> src_array, array_view<float, 2> dest_array, int ksize, float eps = 1500.0f)
	{
		static const int tile_size = 32;
		array_view<float, 2> mean_I(ctx.mean_I);
		array_view<float, 2> mean_p(ctx.mean_p);
		array_view<float, 2> mean_Ip(ctx.mean_Ip);
		array_view<float, 2> mean_II(ctx.mean_II);
		array_view<float, 2> a(ctx.a);
		array_view<float, 2> b(ctx.b);
#if OPTIMIZE_FOR_AMD
		box_filter_32f_c1(ctx.acc_view, guide_array, mean_I, a, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, src_array, mean_p, b, ksize, ksize);
#else
		box_filter_32f_c1(ctx.acc_view, guide_array, mean_I, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, src_array, mean_p, ksize, ksize);
#endif
		detail::box_filter_32f_c1(ctx.acc_view, guide_array, src_array, mean_Ip, a, ksize);
		detail::box_filter_32f_c1(ctx.acc_view, guide_array, guide_array, mean_II, b, ksize);
		parallel_for_each(ctx.acc_view, a.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			float mean_I_val = guarded_read(mean_I, gidx);
			float mean_Ip_val = guarded_read(mean_Ip, gidx);
			float mean_p_val = guarded_read(mean_p, gidx);
			float mean_II_val = guarded_read(mean_II, gidx);
			float cov_Ip_val = mean_Ip_val - mean_I_val * mean_p_val;
			float var_I_val = mean_II_val - mean_I_val * mean_I_val;
			float a_val = cov_Ip_val / (var_I_val + eps);
			float b_val = mean_p_val - a_val * mean_I_val;
			guarded_write(a, gidx, a_val);
			guarded_write(b, gidx, b_val);
		});
#if OPTIMIZE_FOR_AMD
		box_filter_32f_c1(ctx.acc_view, a, mean_I, mean_II, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, b, mean_p, mean_Ip, ksize, ksize);
#else
		box_filter_32f_c1(ctx.acc_view, a, mean_I, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, b, mean_p, ksize, ksize);
#endif
		parallel_for_each(ctx.acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(mean_I, idx.global) * guarded_read(guide_array, idx.global) + guarded_read(mean_p, idx.global));
		});
	}

	inline void guided_filter_32f_c1(guided_filter_context& ctx, array_view<const float, 2> src_array, array_view<float, 2> dest_array, int ksize, float eps = 1500.0f)
	{
		static const int tile_size = 32;
		array_view<float, 2> mean_I(ctx.mean_I);
		array_view<float, 2> mean_p(ctx.mean_p);
		array_view<float, 2> mean_Ip(ctx.mean_Ip);
		array_view<float, 2> mean_II(ctx.mean_II);
		array_view<float, 2> a(ctx.a);
		array_view<float, 2> b(ctx.b);
#if OPTIMIZE_FOR_AMD
		box_filter_32f_c1(ctx.acc_view, src_array, mean_p, b, ksize, ksize);
#else
		box_filter_32f_c1(ctx.acc_view, src_array, mean_p, ksize, ksize);
#endif
		detail::box_filter_32f_c1(ctx.acc_view, src_array, src_array, mean_Ip, a, ksize);
		parallel_for_each(ctx.acc_view, a.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			float mean_Ip_val = guarded_read(mean_Ip, gidx);
			float mean_p_val = guarded_read(mean_p, gidx);
			float cov_Ip_val = mean_Ip_val - mean_p_val * mean_p_val;
			float a_val = cov_Ip_val / (cov_Ip_val + eps);
			float b_val = mean_p_val - a_val * mean_p_val;
			guarded_write(a, gidx, a_val);
			guarded_write(b, gidx, b_val);
		});
#if OPTIMIZE_FOR_AMD
		box_filter_32f_c1(ctx.acc_view, a, mean_I, mean_II, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, b, mean_p, mean_Ip, ksize, ksize);
#else
		box_filter_32f_c1(ctx.acc_view, a, mean_I, ksize, ksize);
		box_filter_32f_c1(ctx.acc_view, b, mean_p, ksize, ksize);
#endif
		parallel_for_each(ctx.acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(mean_I, idx.global) * guarded_read(src_array, idx.global) + guarded_read(mean_p, idx.global));
		});
	}
}
