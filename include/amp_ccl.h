#pragma once

#include "amp_core.h"

namespace amp
{
	namespace detail
	{

		enum class component { left = 1, up = 2, right = 4, down = 8};

		static const int warp_size = 32;
		static const int warp_log = 5;
		static const int cta_size_x = 32;
		static const int cta_size_y = 8;
		static const int sta_size_merge_y = 4;
		static const int sta_size_merge_x = 32;
		static const int tpb_x = 1;
		static const int tpb_y = 4;
		static const int tile_cols = cta_size_x * tpb_x;
		static const int tile_rows = cta_size_y * tpb_y;

		struct connected
		{
			connected(float lo_, float hi_)
				: lo(lo_), hi(hi_)
			{}

			bool operator()(float v1, float v2) const restrict(amp)
			{
				float d = v1 - v2;
				return lo <= d && d <= hi;
			}

			float lo, hi;
		};

		inline void label_tiles(accelerator_view& acc_view, array_view<const int, 2> mask_array, array_view<int, 2> label_array)
		{
			int rows = mask_array.get_extent()[0];
			int cols = mask_array.get_extent()[1];
			concurrency::extent<2> ext(DIVUP(mask_array.get_extent()[0], 4), mask_array.get_extent()[1]);

			parallel_for_each(acc_view, ext.tile<cta_size_y, cta_size_x>().pad(), [=](tiled_index<cta_size_y, cta_size_x> idx) restrict(amp)
			{
				int x = idx.local[1] + idx.tile[1] * tile_cols;
				int y = idx.local[0] + idx.tile[0] * tile_rows;

				tile_static int labelsTile[tile_rows][tile_cols];
				tile_static int edgesTile[tile_rows][tile_cols];
				tile_static int changed;

				int new_labels[tpb_y];
				int old_labels[tpb_y];

				for(int i = 0; i < tpb_y; ++i)
				{
					int yloc = idx.local[0] + cta_size_y * i;
					int xloc = idx.local[1];
					int c = guarded_read(mask_array, concurrency::index<2>(y + cta_size_y * i, x));

					if(!xloc) c &= ~static_cast<int>(component::left);
					if(!yloc) c &= ~static_cast<int>(component::up);
					if(xloc == tile_cols - 1) c &= ~static_cast<int>(component::right);
					if(yloc == tile_rows - 1) c &= ~static_cast<int>(component::down);

					new_labels[i] = yloc * tile_cols + xloc;
					edgesTile[yloc][xloc] = c;
				}

				for(int k = 0;; ++k)
				{
					//1. backup
					old_labels[0] = new_labels[0];
					labelsTile[idx.local[0] + cta_size_y * 0][idx.local[1]] = new_labels[0];
					old_labels[1] = new_labels[1];
					labelsTile[idx.local[0] + cta_size_y * 1][idx.local[1]] = new_labels[1];
					old_labels[2] = new_labels[2];
					labelsTile[idx.local[0] + cta_size_y * 2][idx.local[1]] = new_labels[2];
					old_labels[3] = new_labels[3];
					labelsTile[idx.local[0] + cta_size_y * 3][idx.local[1]] = new_labels[3];
					idx.barrier.wait_with_tile_static_memory_fence();

					//2. compare local arrays
					int yloc = idx.local[0];
					int xloc = idx.local[1];
					int c = edgesTile[yloc][xloc];
					int label = new_labels[0];
					if (c & static_cast<int>(component::up)) label = direct3d::imin(label, labelsTile[yloc - 1][xloc]);
					if (c &  static_cast<int>(component::down)) label = direct3d::imin(label, labelsTile[yloc + 1][xloc]);
					if (c & static_cast<int>(component::left)) label = direct3d::imin(label, labelsTile[yloc][xloc - 1]);
					if (c & static_cast<int>(component::right)) label = direct3d::imin(label, labelsTile[yloc][xloc + 1]);
					new_labels[0] = label;
					yloc += cta_size_y;
					c = edgesTile[yloc][xloc];
					label = new_labels[1];
					if (c & static_cast<int>(component::up)) label = direct3d::imin(label, labelsTile[yloc - 1][xloc]);
					if (c &  static_cast<int>(component::down)) label = direct3d::imin(label, labelsTile[yloc + 1][xloc]);
					if (c & static_cast<int>(component::left)) label = direct3d::imin(label, labelsTile[yloc][xloc - 1]);
					if (c & static_cast<int>(component::right)) label = direct3d::imin(label, labelsTile[yloc][xloc + 1]);
					new_labels[1] = label;
					yloc += cta_size_y;
					c = edgesTile[yloc][xloc];
					label = new_labels[2];
					if (c & static_cast<int>(component::up)) label = direct3d::imin(label, labelsTile[yloc - 1][xloc]);
					if (c &  static_cast<int>(component::down)) label = direct3d::imin(label, labelsTile[yloc + 1][xloc]);
					if (c & static_cast<int>(component::left)) label = direct3d::imin(label, labelsTile[yloc][xloc - 1]);
					if (c & static_cast<int>(component::right)) label = direct3d::imin(label, labelsTile[yloc][xloc + 1]);
					new_labels[2] = label;
					yloc += cta_size_y;
					c = edgesTile[yloc][xloc];
					label = new_labels[3];
					if (c & static_cast<int>(component::up)) label = direct3d::imin(label, labelsTile[yloc - 1][xloc]);
					if (c &  static_cast<int>(component::down)) label = direct3d::imin(label, labelsTile[yloc + 1][xloc]);
					if (c & static_cast<int>(component::left)) label = direct3d::imin(label, labelsTile[yloc][xloc - 1]);
					if (c & static_cast<int>(component::right)) label = direct3d::imin(label, labelsTile[yloc][xloc + 1]);
					new_labels[3] = label;

					changed = 0;
					idx.barrier.wait_with_tile_static_memory_fence();

					//3. determine: Is any value changed?
					int local_changed = 0;
					if (new_labels[0] < old_labels[0])
					{
						local_changed = 1;
						concurrency::atomic_fetch_min(&labelsTile[0][0] + old_labels[0], new_labels[0]);
					}
					if (new_labels[1] < old_labels[1])
					{
						local_changed = 1;
						concurrency::atomic_fetch_min(&labelsTile[0][0] + old_labels[1], new_labels[1]);
					}
					if (new_labels[2] < old_labels[2])
					{
						local_changed = 1;
						concurrency::atomic_fetch_min(&labelsTile[0][0] + old_labels[2], new_labels[2]);
					}
					if (new_labels[3] < old_labels[3])
					{
						local_changed = 1;
						concurrency::atomic_fetch_min(&labelsTile[0][0] + old_labels[3], new_labels[3]);
					}
					concurrency::atomic_fetch_or(&changed, local_changed);
					idx.barrier.wait_with_tile_static_memory_fence();
					if(!changed)
						break;

					//4. Compact paths
					const int *labels = &labelsTile[0][0];
					label = new_labels[0];
					while (labels[label] < label) label = labels[label];
					new_labels[0] = label;
					label = new_labels[1];
					while (labels[label] < label) label = labels[label];
					new_labels[1] = label;
					label = new_labels[2];
					while (labels[label] < label) label = labels[label];
					new_labels[2] = label;
					label = new_labels[3];
					while (labels[label] < label) label = labels[label];
					new_labels[3] = label;
					idx.barrier.wait_with_tile_static_memory_fence();
				}

				for(int i = 0; i < tpb_y; ++i)
				{
					int label = new_labels[i];
					int yloc = label / tile_cols;
					int xloc = label - yloc * tile_cols;

					xloc += idx.tile[1] * tile_cols;
					yloc += idx.tile[0] * tile_rows;

					label = yloc * cols + xloc;
					// do it for x too.
					guarded_write(label_array, concurrency::index<2>(y + cta_size_y * i, x), label);
				}
			});
		}

		inline int root(array_view<const int, 2> label_array, int label) restrict(amp)
		{
			while (1)
			{
				int y = label / label_array.get_extent()[1];
				int x = label - y * label_array.get_extent()[1];

				int parent = label_array(y, x);

				if (label == parent) break;

				label = parent;
			}
			return label;
		}

		inline void is_connected(array_view<int, 2> label_array, int l1, int l2, int& changed) restrict(amp)
		{
			int r1 = root(label_array, l1);
			int r2 = root(label_array, l2);

			if (r1 == r2) return;

			int mi = direct3d::imin(r1, r2);
			int ma = direct3d::imax(r1, r2);

			int y = ma / label_array.get_extent()[1];
			int x = ma - y * label_array.get_extent()[1];

			concurrency::atomic_fetch_min(&label_array(y, x), mi);
			changed = 1;
		}

		inline void cross_merge(accelerator_view& acc_view, concurrency::extent<2> ext, const int tiles_num_y, const int tiles_num_x, int tile_size_y_, int tile_size_x_,
			array_view<const int, 2> mask_array, array_view<int, 2> label_array, const int y_incomplete, int x_incomplete)
		{
			parallel_for_each(acc_view, ext.tile<sta_size_merge_y, sta_size_merge_x>(), [=](tiled_index<sta_size_merge_y, sta_size_merge_x> idx) restrict(amp)
			{
				tile_static int shared_changed;

				int tile_size_y = tile_size_y_;
				int tile_size_x = tile_size_x_;

				int tid = idx.local[0] * idx.tile_dim1 + idx.local[1];
				int stride = idx.tile_dim0 * idx.tile_dim1;

				int ybegin = idx.tile[0] * (tiles_num_y * tile_size_y);
				int yend = ybegin + tiles_num_y * tile_size_y;
				int xbegin = idx.tile[1] * (tiles_num_x * tile_size_x);
				int xend = xbegin + tiles_num_x * tile_size_x;
				int tasksV = (tiles_num_x - 1) * (yend - ybegin);
				int tasksH = (tiles_num_y - 1) * (xend - xbegin);
				int total = tasksH + tasksV;
				do
				{
					shared_changed = 0;
					idx.barrier.wait_with_tile_static_memory_fence();
					int changed = 0;
					for (int taskIdx = tid; taskIdx < total; taskIdx += stride)
					{
						if (taskIdx < tasksH)
						{
							int indexH = taskIdx;
							int row = indexH / (xend - xbegin);
							int col = indexH - row * (xend - xbegin);
							int y = ybegin + (row + 1) * tile_size_y;
							int x = xbegin + col;
							int e = mask_array(y, x);
							if (e & static_cast<int>(component::up))
							{
								int lc = label_array(y, x);
								int lu = label_array(y - 1, x);
								is_connected(label_array, lc, lu, changed);
							}
						}
						else
						{
							int indexV = taskIdx - tasksH;
							int col = indexV / (yend - ybegin);
							int row = indexV - col * (yend - ybegin);
							int x = xbegin + (col + 1) * tile_size_x;
							int y = ybegin + row;
							int e = mask_array(y, x);
							if (e & static_cast<int>(component::left))
							{
								int lc = label_array(y, x);
								int ll = label_array(y, x - 1);
								is_connected(label_array, lc, ll, changed);
							}
						}
					}
					concurrency::atomic_fetch_or(&shared_changed, changed);
					idx.barrier.wait_with_tile_static_memory_fence();
				} while (shared_changed);
			});
		}

		inline void flattern(accelerator_view& acc_view, array_view<const int, 2> mask_array, array_view<int, 2> label_array)
		{
			parallel_for_each(acc_view, label_array.get_extent().tile<8, 32>().pad(), [=](tiled_index<8, 32> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				if (label_array.get_extent().contains(gidx))
				{
					label_array(gidx) = root(label_array, label_array(gidx));
				}
			});
		}
	}

	inline void compute_connectivity_mask(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<int, 2> mask_array, float lo, float hi)
	{
		detail::connected connected(lo, hi);
		parallel_for_each(acc_view, src_array.get_extent().tile<detail::cta_size_y, detail::cta_size_x>().pad(), [=](tiled_index<detail::cta_size_y, detail::cta_size_x> idx) restrict(amp)
		{
			float intensity = guarded_read(src_array, idx.global);
			int c = 0;
			int x = idx.global[1];
			int y = idx.global[0];
			if (x > 0 && connected(intensity, guarded_read(src_array, concurrency::index<2>(y, x - 1), FLT_MAX)))
				c |= static_cast<int>(detail::component::left);

			if (y > 0 && connected(intensity, guarded_read(src_array, concurrency::index<2>(y - 1, x), FLT_MAX)))
				c |= static_cast<int>(detail::component::up);

			if (x + 1 < src_array.get_extent()[1] && connected(intensity, guarded_read(src_array, concurrency::index<2>(y, x + 1), FLT_MAX)))
				c |= static_cast<int>(detail::component::right);

			if (y + 1 < src_array.get_extent()[0] && connected(intensity, guarded_read(src_array, concurrency::index<2>(y + 1, x), FLT_MAX)))
				c |= static_cast<int>(detail::component::down);
			guarded_write(mask_array, idx.global, c);
		});
	}

	inline void label_components(accelerator_view& acc_view, array_view<const int, 2> mask_array, array_view<int, 2> label_array)
	{
		concurrency::extent<2> grid(DIVUP(mask_array.get_extent()[0], detail::tile_rows), DIVUP(mask_array.get_extent()[1], detail::tile_cols));
		detail::label_tiles(acc_view, mask_array, label_array);
		int tile_size_x = detail::tile_cols, tile_size_y = detail::tile_rows;
		while (grid[0] > 1 || grid[1] > 1)
		{
			concurrency::extent<2> ext((int)ceilf(grid[0] / 2.f) * detail::sta_size_merge_y, (int)ceilf(grid[1] / 2.f) * detail::sta_size_merge_x);
			detail::cross_merge(acc_view, ext, 2, 2, tile_size_y, tile_size_x, mask_array, label_array
				, (int)ceilf(grid[0] / 2.f) - grid[0] / 2, (int)ceilf(grid[1] / 2.f) - grid[1] / 2);
			tile_size_x <<= 1;
			tile_size_y <<= 1;
			grid[0] = (int)ceilf(grid[0] / 2.f);
			grid[1] = (int)ceilf(grid[1] / 2.f);
		}
		detail::flattern(acc_view, mask_array, label_array);
	}

	// set labels of all zero pixels to -1
	inline void binary_filter_labels(accelerator_view& acc_view, array_view<const float, 2> src_array, array_view<int, 2> label_array)
	{
		static const int tile_size = 32;
		parallel_for_each(acc_view, src_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			float src_val = guarded_read(src_array, idx.global);
			int label_val = guarded_read(label_array, idx.global);
			guarded_write(label_array, idx.global, src_val == 0.0f ? -1 : label_val);
		});
	}

	// reorder labels to 0...N-1(N is the number of unique labels) and return N
	template<unsigned int MaxReservedLabels = 1024U>
	inline int reorder_labels(accelerator_view& acc_view, array_view<int, 2> label_array)
	{
		static const int tile_size = 32;
		// Copy labels to CPU
		std::vector<int> cpu_label_array(label_array.get_extent().size());
		concurrency::copy(label_array, &cpu_label_array[0]);
		// Sort labels
		std::sort(cpu_label_array.begin(), cpu_label_array.end());
		// Labels
		std::vector<int> unique_label_array;
		unique_label_array.reserve(MaxReservedLabels);
		int current_label = -1;
		for (size_t i = 0; i < cpu_label_array.size(); i++)
		{
			int label = cpu_label_array[i];
			if (label != current_label)
			{
				unique_label_array.emplace_back(label);
				current_label = label;
			}
		}
		// Upload to GPU
		if (unique_label_array.size() > MaxReservedLabels)
		{
			throw std::runtime_error("Too many unique labels for reorder_labels");
		}
		kernel_wrapper<int, MaxReservedLabels> unique_labels(unique_label_array.begin(), unique_label_array.end());
		int unique_label_count = int(unique_label_array.size());
		// Filter Image on GPU
		concurrency::parallel_for_each(acc_view, label_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			int label_value = guarded_read(label_array, idx.global);
			int start_index = 0;
			int end_index = unique_label_count;
			int mid_index = (start_index + end_index) / 2;
			do
			{
				int mid_value = unique_labels.data[mid_index];
				if (mid_value == label_value)
				{
					break;
				}
				else if (label_value < mid_value)
				{
					end_index = mid_index;
				}
				else
				{
					start_index = mid_index + 1;
				}
				mid_index = (start_index + end_index) / 2;
			} while (start_index < end_index);
			guarded_write(label_array, idx.global, mid_index);
		});
		return unique_label_count;
	}

}
