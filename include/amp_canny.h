#pragma once

#include "amp_core.h"

namespace amp
{
	// Canny edge detection(only support aperture size 3)
	class canny_context
	{
	public:
		canny_context(const accelerator_view& acc_view_, int rows_, int cols_)
			: acc_view(acc_view_), dx(rows_, cols_, acc_view_), dy(rows_, cols_, acc_view_)
			, dx_buf(rows_, cols_, acc_view_), dy_buf(rows_, cols_, acc_view_)
			, mag_buf(rows_ + 2, cols_ + 2, acc_view_), map_buf(rows_ + 2, cols_ + 2, acc_view_)
			, track_buf1(rows_ * cols_, acc_view_), track_buf2(rows_ * cols_, acc_view_), counter(1, acc_view_)
		{
		}

		concurrency::array<int, 2> dx, dy, dx_buf, dy_buf;
		concurrency::array<float, 2> mag_buf;
		concurrency::array<int, 2> map_buf;
		concurrency::array<unsigned int, 1> track_buf1, track_buf2;
		concurrency::array<unsigned int, 1> counter;
		accelerator_view acc_view;
	};

	namespace detail
	{
		inline void canny_sobel_row_pass(canny_context& ctx, array_view<const float, 2> src_array)
		{
			static const int tile_size = 32;
			array_view<int, 2> dx_buf(ctx.dx_buf);
			array_view<int, 2> dy_buf(ctx.dy_buf);
			dx_buf.discard_data();
			dy_buf.discard_data();
			concurrency::parallel_for_each(ctx.acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int gidx = idx.global[1];
				int gidy = idx.global[0];

				int lidx = idx.local[1];
				int lidy = idx.local[0];

				int rows = src_array.get_extent()[0];
				int cols = src_array.get_extent()[1];

				tile_static int smem[tile_size][tile_size + 2];
				smem[lidy][lidx + 1] = int(fast_math::roundf(src_array(direct3d::imin(gidy, rows - 1), gidx)));
				if(lidx == 0)
				{
					smem[lidy][0] = int(fast_math::roundf(src_array(direct3d::imin(gidy, rows - 1), direct3d::imax(gidx - 1, 0))));
					smem[lidy][tile_size + 1] = int(fast_math::roundf(src_array(direct3d::imin(gidy, rows - 1), direct3d::imin(gidx + tile_size, cols - 1))));
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				guarded_write(dx_buf, idx.global, -smem[lidy][lidx] + smem[lidy][lidx + 2]);
				guarded_write(dy_buf, idx.global, smem[lidy][lidx] + 2 * smem[lidy][lidx + 1] + smem[lidy][lidx + 2]);
			});
		}

		inline void canny_magnitude(canny_context& ctx, bool l2_grad)
		{
			static const int tile_size = 32;
			array_view<const int, 2> dx_buf(ctx.dx_buf);
			array_view<const int, 2> dy_buf(ctx.dy_buf);
			array_view<int, 2> dx(ctx.dx);
			array_view<int, 2> dy(ctx.dy);
			array_view<float, 2> mag_buf(ctx.mag_buf);
			mag_buf.discard_data();
			dx.discard_data();
			dy.discard_data();
			concurrency::extent<2> src_extent = dx.get_extent();
			concurrency::parallel_for_each(ctx.acc_view, src_extent.tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int gidx = idx.global[1];
				int gidy = idx.global[0];

				int lidx = idx.local[1];
				int lidy = idx.local[0];

				int rows = src_extent[0];
				int cols = src_extent[1];

				tile_static int sdx[tile_size + 2][tile_size];
				tile_static int sdy[tile_size + 2][tile_size];

				sdx[lidy + 1][lidx] = dx_buf(direct3d::imin(gidy, rows - 1), gidx);
				sdy[lidy + 1][lidx] = dy_buf(direct3d::imin(gidy, rows - 1), gidx);
				if(lidy == 0)
				{
					sdx[0][lidx] = dx_buf(direct3d::imin(direct3d::imax(gidy - 1, 0), rows - 1), gidx);
					sdx[tile_size + 1][lidx] = dx_buf(direct3d::imin(gidy + tile_size, rows - 1), gidx);

					sdy[0][lidx] = dy_buf(direct3d::imin(direct3d::imax(gidy - 1, 0), rows - 1), gidx);
					sdy[tile_size + 1][lidx] = dy_buf(direct3d::imin(gidy + tile_size, rows - 1), gidx);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				int x = sdx[lidy][lidx] + 2 * sdx[lidy + 1][lidx] + sdx[lidy + 2][lidx];
				int y = -sdy[lidy][lidx] + sdy[lidy + 2][lidx];
				if(gidx < cols && gidy < rows)
				{
					dx(idx.global) = x;
					dy(idx.global) = y;
					mag_buf(concurrency::index<2>(gidy + 1, gidx + 1)) = l2_grad ? fast_math::sqrtf(float(x * x + y * y)) : float(direct3d::abs(x) + direct3d::abs(y));
				}
			});
		}

		inline void canny_map(canny_context& ctx, float low_thresh, float high_thresh)
		{
			static const int tile_size = 32;
			static const int canny_shift = 15;
			static const int tg22 = (int)(0.4142135623730950488016887242097*(1 << canny_shift) + 0.5);
			array_view<const int, 2> dx(ctx.dx);
			array_view<const int, 2> dy(ctx.dy);
			array_view<const float, 2> mag_buf(ctx.mag_buf);
			array_view<int, 2> map_buf(ctx.map_buf);
			map_buf.discard_data();

			concurrency::parallel_for_each(ctx.acc_view, dx.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int gidx = idx.global[1];
				int gidy = idx.global[0];

				int lidx = idx.local[1];
				int lidy = idx.local[0];

				int grp_idx = idx.tile_origin[1];
				int grp_idy = idx.tile_origin[0];

				int rows = dx.get_extent()[0];
				int cols = dx.get_extent()[1];

				tile_static float smem[tile_size + 2][tile_size + 2];
				int tid = lidx + lidy * tile_size;
				int lx = tid % (tile_size + 2);
				int ly = tid / (tile_size + 2);
				if(ly < tile_size - 2)
				{
					smem[ly][lx] = mag_buf(direct3d::imin(grp_idy + ly, rows - 1), grp_idx + lx);
				}
				if(ly < 4 && grp_idy + ly + tile_size - 2 <= rows && grp_idx + lx <= cols)
				{
					smem[ly + tile_size - 2][lx] = mag_buf(direct3d::imin(grp_idy + ly + tile_size - 2, rows - 1), grp_idx + lx);
				}
				idx.barrier.wait_with_tile_static_memory_fence();

				if(gidy < rows && gidx < cols)
				{
					int x = dx(idx.global);
					int y = dy(idx.global);
					const int s = (x ^ y) < 0 ? -1 : 1;
					const float m = smem[lidy + 1][lidx + 1];
					x = direct3d::abs(x);
					y = direct3d::abs(y);

					// 0 - the pixel can not belong to an edge
					// 1 - the pixel might belong to an edge
					// 2 - the pixel does belong to an edge
					int edge_type = 0;
					if(m > low_thresh)
					{
						const int tg22x = x * tg22;
						const int tg67x = tg22x + (x << (1 + canny_shift));
						y <<= canny_shift;
						if(y < tg22x)
						{
							if(m > smem[lidy + 1][lidx] && m >= smem[lidy + 1][lidx + 2])
							{
								edge_type = 1 + (int)(m > high_thresh);
							}
						}
						else if(y > tg67x)
						{
							if(m > smem[lidy][lidx + 1] && m >= smem[lidy + 2][lidx + 1])
							{
								edge_type = 1 + (int)(m > high_thresh);
							}
						}
						else
						{
							if(m > smem[lidy][lidx + 1 - s] && m > smem[lidy + 2][lidx + 1 + s])
							{
								edge_type = 1 + (int)(m > high_thresh);
							}
						}
					}
					map_buf(gidy + 1, gidx + 1) = edge_type;
				}
			});
		}

		inline void canny_edge_hysteresis_local(canny_context& ctx)
		{
			static const int tile_size = 16;
			array_view<int, 2> map_buf(ctx.map_buf);
			array_view<unsigned int, 1> counter(ctx.counter);
			counter.discard_data();
			concurrency::parallel_for_each(ctx.acc_view, ctx.counter.get_extent(), [=](concurrency::index<1> idx) restrict(amp)
			{
				counter(idx) = 0u;
			});
			array_view<unsigned int, 1> st(ctx.track_buf1);
			st.discard_data();
			concurrency::parallel_for_each(ctx.acc_view, ctx.dx.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				tile_static int smem[tile_size + 2][tile_size + 2];

				int_2 blockDim(idx.tile_dim1, idx.tile_dim0);
				int_2 threadIdx(idx.local[1], idx.local[0]);

				const int x = idx.global[1];
				const int y = idx.global[0];

				smem[threadIdx.y + 1][threadIdx.x + 1] = guarded_read(map_buf, concurrency::index<2>(y + 1, x + 1));
				if(threadIdx.y == 0)
					smem[0][threadIdx.x + 1] = guarded_read(map_buf, concurrency::index<2>(y, x + 1));
				if(threadIdx.y == blockDim.y - 1)
					smem[blockDim.y + 1][threadIdx.x + 1] = guarded_read(map_buf, concurrency::index<2>(y + 2, x + 1));
				if(threadIdx.x == 0)
					smem[threadIdx.y + 1][0] = guarded_read(map_buf, concurrency::index<2>(y + 1, x));
				if(threadIdx.x == blockDim.x - 1)
					smem[threadIdx.y + 1][blockDim.x + 1] = guarded_read(map_buf, concurrency::index<2>(y + 1, x + 2));
				if(threadIdx.x == 0 && threadIdx.y == 0)
					smem[0][0] = guarded_read(map_buf, concurrency::index<2>(y, x));
				if(threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
					smem[0][blockDim.x + 1] = guarded_read(map_buf, concurrency::index<2>(y, x + 2));
				if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
					smem[blockDim.y + 1][0] = guarded_read(map_buf, concurrency::index<2>(y + 2, x));
				if(threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
					smem[blockDim.y + 1][blockDim.x + 1] = guarded_read(map_buf, concurrency::index<2>(y + 2, x + 2));
				idx.barrier.wait_with_tile_static_memory_fence();

				int n;
				// TODO unroll
				for(int k = 0; k < tile_size; ++k)
				{
					n = 0;
					if(smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
					{
						n += smem[threadIdx.y][threadIdx.x] == 2;
						n += smem[threadIdx.y][threadIdx.x + 1] == 2;
						n += smem[threadIdx.y][threadIdx.x + 2] == 2;
						n += smem[threadIdx.y + 1][threadIdx.x] == 2;
						n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;
						n += smem[threadIdx.y + 2][threadIdx.x] == 2;
						n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
						n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
					}
					if(n > 0)
						smem[threadIdx.y + 1][threadIdx.x + 1] = 2;
					idx.barrier.wait_with_tile_static_memory_fence();
				}
				const int e = smem[threadIdx.y + 1][threadIdx.x + 1];
				guarded_write(map_buf, concurrency::index<2>(y + 1, x + 1), e);
				n = 0;
				if(e == 2)
				{
					n += smem[threadIdx.y][threadIdx.x] == 1;
					n += smem[threadIdx.y][threadIdx.x + 1] == 1;
					n += smem[threadIdx.y][threadIdx.x + 2] == 1;

					n += smem[threadIdx.y + 1][threadIdx.x] == 1;
					n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;

					n += smem[threadIdx.y + 2][threadIdx.x] == 1;
					n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
					n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
				}
				if(n > 0)
				{
					const unsigned int ind = concurrency::atomic_fetch_inc(&counter[0]);
					st[ind] = (unsigned int(y + 1) << 16) + unsigned int(x + 1);
				}
			});
		}

		struct c_dx_dy_wrapper
		{
			int dx[8];
			int dy[8];

			c_dx_dy_wrapper()
			{
				std::vector<int> temp_dx{ -1, 0, 1, -1, 1, -1, 0, 1 };
				std::vector<int> temp_dy{ -1, -1, -1, 0, 0, 1, 1, 1 };
				memcpy(dx, &temp_dx[0], sizeof(int) * 8);
				memcpy(dy, &temp_dy[0], sizeof(int) * 8);
			}
		};

		inline void canny_edge_hysteresis_global(canny_context& ctx)
		{
			static const int tile_size = 128;
			static const int stack_size = 512;
			array_view<int, 2> map_buf(ctx.map_buf);
			array_view<unsigned int, 1> counter(ctx.counter);
			while(true)
			{
				unsigned int prev_counter = 0u;
				concurrency::copy(counter, &prev_counter);
				if(prev_counter == 0u)
				{
					break;
				}
				concurrency::parallel_for_each(ctx.acc_view, ctx.counter.get_extent(), [=](concurrency::index<1> idx) restrict(amp)
				{
					counter(idx) = 0u;
				});
				array_view<unsigned int, 1> st1(ctx.track_buf1);
				array_view<unsigned int, 1> st2(ctx.track_buf2);
				concurrency::extent<2> dest_extent = ctx.dx.get_extent();
				concurrency::extent<2> kernel_extent((prev_counter + 65534) / 65535, std::min<unsigned int>(prev_counter, 65535u) * 128);
				c_dx_dy_wrapper c;
				concurrency::parallel_for_each(ctx.acc_view, kernel_extent.tile<1, tile_size>().pad(), [=](tiled_index<1, tile_size> idx) restrict(amp)
				{
					int lidx = idx.local[1];
					int grp_idx = idx.tile[1];
					int grp_idy = idx.tile[0];
					int cols = dest_extent[1];
					int rows = dest_extent[0];

					tile_static unsigned int s_counter;
					tile_static unsigned int s_ind;
					tile_static graphics::uint_2 s_st[stack_size];

					if(lidx == 0)
					{
						s_counter = 0;
					}
					idx.barrier.wait_with_tile_static_memory_fence();

					unsigned int ind = direct3d::mad(grp_idy, idx.tile_dim1, grp_idx);
					if(ind < prev_counter)
					{
						unsigned int pos0 = st1[ind];
						graphics::uint_2 pos(pos0 & 0xffff, pos0 >> 16);
						if(lidx < 8)
						{
							pos.x += c.dx[lidx];
							pos.y += c.dy[lidx];
							if(guarded_read(map_buf, concurrency::index<2>(pos.y, pos.x)) == 1)
							{
								guarded_write(map_buf, concurrency::index<2>(pos.y, pos.x), 2);
								ind = concurrency::atomic_fetch_inc(&s_counter);
								s_st[ind] = pos;
							}
						}
						idx.barrier.wait_with_tile_static_memory_fence();

						while(s_counter > 0 && s_counter <= stack_size - idx.tile_dim1)
						{
							const int subTaskIdx = lidx >> 3;
							const int portion = direct3d::umin(s_counter, (unsigned int)(idx.tile_dim1 >> 3));
							if(subTaskIdx < portion)
								pos = s_st[s_counter - 1 - subTaskIdx];
							idx.barrier.wait_with_tile_static_memory_fence();
							if(lidx == 0)
								s_counter -= portion;
							idx.barrier.wait_with_tile_static_memory_fence();
							if(subTaskIdx < portion)
							{
								pos.x += c.dx[lidx & 7];
								pos.y += c.dy[lidx & 7];
								if(guarded_read(map_buf, concurrency::index<2>(pos.y, pos.x)) == 1)
								{
									guarded_write(map_buf, concurrency::index<2>(pos.y, pos.x), 2);
									ind = concurrency::atomic_fetch_inc(&s_counter);
									s_st[ind] = pos;
								}
							}
							idx.barrier.wait_with_tile_static_memory_fence();
						}
						if(s_counter > 0)
						{
							if(lidx == 0)
							{
								ind = concurrency::atomic_fetch_add(&counter[0], s_counter);
								s_ind = ind - s_counter;
							}
							idx.barrier.wait_with_tile_static_memory_fence();
							ind = s_ind;
							for(int i = lidx; i < (int)s_counter; i += idx.tile_dim1)
							{
								st2[ind + i] = (s_st[i].y << 16) + s_st[i].x;
							}
						}
					}
				});
				std::swap(ctx.track_buf1, ctx.track_buf2);
			}
		}

		inline void canny_get_edges(canny_context& ctx, array_view<float, 2> dest_array)
		{
			static const int tile_size = 32;
			array_view<int, 2> map_buf(ctx.map_buf);
			concurrency::parallel_for_each(ctx.acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int gidx = idx.global[1];
				int gidy = idx.global[0];
				int map_value = guarded_read(map_buf, concurrency::index<2>(gidy + 1, gidx + 1));
				guarded_write(dest_array, idx.global, map_value == 2 ? 255.0f : 0.0f);
			});
		}
	}

	inline void canny_32f_c1(canny_context& ctx, array_view<const float, 2> src_array, array_view<float, 2> dest_array
		, float low_thresh, float high_thresh, bool l2_grad = false)
	{
		if(low_thresh > high_thresh)
			std::swap(low_thresh, high_thresh);
		static const int tile_size = 32;
		// fill 0.0f
		array_view<float, 2> mag_buf(ctx.mag_buf);
		array_view<int, 2> map_buf(ctx.map_buf);
		dest_array.discard_data();
		mag_buf.discard_data();
		map_buf.discard_data();
		concurrency::parallel_for_each(ctx.acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, 0.0f);
		});
		concurrency::parallel_for_each(ctx.acc_view, mag_buf.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(mag_buf, idx.global, 0.0f);
			guarded_write(map_buf, idx.global, 0);
		});
		detail::canny_sobel_row_pass(ctx, src_array);
		detail::canny_magnitude(ctx, l2_grad);
		detail::canny_map(ctx, low_thresh, high_thresh);
		detail::canny_edge_hysteresis_local(ctx);
		detail::canny_edge_hysteresis_global(ctx);
		detail::canny_get_edges(ctx, dest_array);
	}
	
}
