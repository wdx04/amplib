#pragma once

#include "amp_core.h"

namespace amp
{
#ifdef _DEBUG
#define MAX_BILATERAL_D 25
#define MAX_BILATERAL_WEIGHT_SIZE 442U
#else
#define MAX_BILATERAL_D 35
#define MAX_BILATERAL_WEIGHT_SIZE 902U
#endif

	inline void bilateral_filter_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, int d, float sigma_color, float sigma_space)
	{
		static const int tile_size = 32;
		assert(d <= MAX_BILATERAL_D);
		int dst_rows = dest_array.get_extent()[0];
		int dst_cols = dest_array.get_extent()[1];
		int i, j, maxk, radius;
		if(sigma_color <= 0.0f)
			sigma_color = 1.0f;
		if(sigma_space <= 0.0f)
			sigma_space = 1.0f;
		float gauss_color_coeff = -0.5f / (sigma_color * sigma_color);
		float gauss_space_coeff = -0.5f / (sigma_space * sigma_space);
		if(d <= 0)
			radius = cvRound(sigma_space * 1.5f);
		else
			radius = d / 2;
		radius = std::max<int>(radius, 1);
		d = radius * 2 + 1;
		kernel_wrapper<float, MAX_BILATERAL_WEIGHT_SIZE> weights;
		kernel_wrapper<int, MAX_BILATERAL_WEIGHT_SIZE> xofs;
		kernel_wrapper<int, MAX_BILATERAL_WEIGHT_SIZE> yofs;
		for(i = -radius, maxk = 0; i <= radius; i++)
		{
			for(j = -radius; j <= radius; j++)
			{
				double r = std::sqrt((double)i * i + (double)j * j);
				if(r > radius)
					continue;
				weights.data[maxk] = (float)std::exp(r * r * gauss_space_coeff);
				yofs.data[maxk] = i;
				xofs.data[maxk] = j;
				maxk++;
			}
		}
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int x = idx.global[1];
			int y = idx.global[0];

			if(y < dst_rows && x < dst_cols)
			{
				float_t sum = 0.0f;
				float wsum = 0.0f;
				float val0 = src_array(idx);
				for(int k = 0; k < maxk; k++)
				{
					float val = guarded_read_reflect101(src_array, concurrency::index<2>(y + yofs.data[k], x + xofs.data[k]));
					float diff = fast_math::fabsf(val - val0);
					float w = weights.data[k] * fast_math::expf(diff * diff * gauss_color_coeff);
					sum += val * w;
					wsum += w;
				}
				dest_array(idx) = sum / wsum;
			}
		});
	}
}
