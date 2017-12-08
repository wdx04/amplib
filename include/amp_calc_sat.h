#pragma once

#include "amp_core.h"

namespace amp
{
	// calculate SAT
#define CALC_SAT_ROW_PASS(i) sum += src_array(row, col + i); \
	row_temp_array(row, col + i) = sum
#define CALC_SAT_COL_PASS(i) sum += row_temp_array(row + i, col); \
	dest_array(row + i, col) = sum
	inline void calc_sat_32f_c1(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<float, 2> row_temp_array)
	{
		static const int tile_size = 16;
		int rows = dest_array.get_extent()[0];
		int cols = dest_array.get_extent()[1];
		int rows_rounddown = ROUNDDOWN(rows, 16);
		int cols_rounddown = ROUNDDOWN(cols, 16);
		// horizontal pass
		row_temp_array.discard_data();
		concurrency::parallel_for_each(acc_view, concurrency::extent<1>(rows).tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
		{
			float sum = 0.0f;
			int row = idx.global[0];
			if(row < rows)
			{
				int col = 0;
				for(; col < cols_rounddown; col += 16)
				{
					CALC_SAT_ROW_PASS(0);
					CALC_SAT_ROW_PASS(1);
					CALC_SAT_ROW_PASS(2);
					CALC_SAT_ROW_PASS(3);
					CALC_SAT_ROW_PASS(4);
					CALC_SAT_ROW_PASS(5);
					CALC_SAT_ROW_PASS(6);
					CALC_SAT_ROW_PASS(7);
					CALC_SAT_ROW_PASS(8);
					CALC_SAT_ROW_PASS(9);
					CALC_SAT_ROW_PASS(10);
					CALC_SAT_ROW_PASS(11);
					CALC_SAT_ROW_PASS(12);
					CALC_SAT_ROW_PASS(13);
					CALC_SAT_ROW_PASS(14);
					CALC_SAT_ROW_PASS(15);
				}
				for(; col < cols; col++)
				{
					sum += src_array(row, col);
					row_temp_array(row, col) = sum;
				}
			}
		});
		// vertical pass
		dest_array.discard_data();
		concurrency::parallel_for_each(acc_view, concurrency::extent<1>(cols).tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
		{
			float sum = 0.0f;
			int col = idx.global[0];
			if(col < cols)
			{
				int row = 0;
				for(; row < rows_rounddown; row += 16)
				{
					CALC_SAT_COL_PASS(0);
					CALC_SAT_COL_PASS(1);
					CALC_SAT_COL_PASS(2);
					CALC_SAT_COL_PASS(3);
					CALC_SAT_COL_PASS(4);
					CALC_SAT_COL_PASS(5);
					CALC_SAT_COL_PASS(6);
					CALC_SAT_COL_PASS(7);
					CALC_SAT_COL_PASS(8);
					CALC_SAT_COL_PASS(9);
					CALC_SAT_COL_PASS(10);
					CALC_SAT_COL_PASS(11);
					CALC_SAT_COL_PASS(12);
					CALC_SAT_COL_PASS(13);
					CALC_SAT_COL_PASS(14);
					CALC_SAT_COL_PASS(15);
				}
				for(; row < rows; row++)
				{
					sum += row_temp_array(row, col);
					dest_array(row, col) = sum;
				}
			}
		});
	}
#undef CALC_SAT_ROW_PASS
#undef CALC_SAT_COL_PASS
}
