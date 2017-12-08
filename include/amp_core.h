#pragma once

#include <cassert>
#include <exception>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#ifdef _MSC_VER
#include <amp_graphics.h>
#include <d3d11.h>
#pragma comment(lib, "d3d11.lib")
#endif
#include <opencv2/core/core.hpp>

#define OPTIMIZE_FOR_AMD 1
#define OPTIMIZE_FOR_INTEL 0

#define DIVUP(total, grain) ((total + grain - 1) / (grain))
#define DIVDOWN(total, grain) ((total) / (grain))
#define ROUNDDOWN(x, s) ((x) & ~((s)-1))
#define ROUNDUP(x, s) ((x+s-1) & ~((s)-1))

namespace std
{
	template<typename T1, typename T2>
	static ostream& operator<<(ostream& os, const std::pair<T1, T2>& v)
	{
		os << v.first << ',' << v.second;
		return os;
	}
}

#ifndef _MSC_VER
// Emulation of HLSL intrinsics
namespace Concurrency
{
	namespace direct3d
	{
		inline float clamp(float v, float lo, float hi) restrict(cpu, amp)
		{
			return v < lo ? lo : (v < hi ? v : hi);
		}

		inline int clamp(int v, int lo, int hi) restrict(cpu, amp)
		{
			return v < lo ? lo : (v < hi ? v : hi);
		}

		inline float mad(float x, float y, float z) restrict(cpu, amp)
		{
			return x * y + z;
		}

		inline int imin(int x, int y) restrict(cpu, amp)
		{
			return x < y ? x : y;
		}

		inline int imax(int x, int y) restrict(cpu, amp)
		{
			return x > y ? x : y;
		}

		inline int abs(int x) restrict(cpu, amp)
		{
			return x < 0 ? -x : x;
		}

		inline int mad(int x, int y, int z) restrict(cpu, amp)
		{
			return x * y + z;
		}
	}
}
#endif

namespace amp
{
	namespace direct3d = concurrency::direct3d;
	namespace fast_math = concurrency::fast_math;
	namespace graphics = concurrency::graphics;

	using concurrency::accelerator_view;
	using concurrency::array_view;
	using concurrency::tiled_index;
	using graphics::float_2;
	using graphics::float_3;
	using graphics::float_4;
	using graphics::int_2;
	using graphics::int_3;
	using graphics::int_4;

	// Helper Functions
#ifdef _WIN32
	inline accelerator_view create_default_hardware_view()
	{
		std::array<D3D_FEATURE_LEVEL, 2> featureLevels = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1 };
		ID3D11Device *device = nullptr;
		HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, featureLevels.data()
			, UINT(featureLevels.size()), D3D11_SDK_VERSION, &device, nullptr, nullptr);
		if(SUCCEEDED(hr))
		{
			accelerator_view acc_view = concurrency::direct3d::create_accelerator_view(device);
			device->Release();
			return acc_view;
		}
		else
		{
			return concurrency::accelerator().create_view();
		}
	}
#endif

	template <typename value_type, int rank>
	inline value_type guarded_read(const array_view<const value_type, rank>& A, const concurrency::index<rank>& idx) restrict(cpu, amp)
	{
		return A.get_extent().contains(idx) ? A[idx] : value_type();
	}

	template <typename value_type, int rank>
	inline value_type guarded_read(const array_view<const value_type, rank>& A, const concurrency::index<rank>& idx, value_type default_value) restrict(cpu, amp)
	{
		return A.get_extent().contains(idx) ? A[idx] : default_value;
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101_for_row(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_col = idx[1] < 0 ? -idx[1] : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] * 2 - idx[1] - 2 : idx[1]);
		return A(idx[0], ref_col);
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101_for_col(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_row = idx[0] < 0 ? -idx[0] : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] * 2 - idx[0] - 2 : idx[0]);
		return A(ref_row, idx[1]);
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_row = idx[0] < 0 ? -idx[0] : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] * 2 - idx[0] - 2 : idx[0]);
		int ref_col = idx[1] < 0 ? -idx[1] : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] * 2 - idx[1] - 2 : idx[1]);
		return A(ref_row, ref_col);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate_for_row(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_col = idx[1] < 0 ? 0 : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] - 1 : idx[1]);
		return A(idx[0], rep_col);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate_for_col(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_row = idx[0] < 0 ? 0 : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] - 1 : idx[0]);
		return A(rep_row, idx[1]);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate(const array_view<const value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_row = idx[0] < 0 ? 0 : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] - 1 : idx[0]);
		int rep_col = idx[1] < 0 ? 0 : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] - 1 : idx[1]);
		return A(rep_row, rep_col);
	}

	template <typename value_type, int rank>
	inline value_type guarded_read(const array_view<value_type, rank>& A, const concurrency::index<rank>& idx) restrict(cpu, amp)
	{
		return A.get_extent().contains(idx) ? A[idx] : value_type();
	}

	template <typename value_type, int rank>
	inline value_type guarded_read(const array_view<value_type, rank>& A, const concurrency::index<rank>& idx, value_type default_value) restrict(cpu, amp)
	{
		return A.get_extent().contains(idx) ? A[idx] : default_value;
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101_for_row(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_col = idx[1] < 0 ? -idx[1] : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] * 2 - idx[1] - 2 : idx[1]);
		return A(idx[0], ref_col);
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101_for_col(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_row = idx[0] < 0 ? -idx[0] : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] * 2 - idx[0] - 2 : idx[0]);
		return A(ref_row, idx[1]);
	}

	template <typename value_type>
	inline value_type guarded_read_reflect101(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int ref_row = idx[0] < 0 ? -idx[0] : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] * 2 - idx[0] - 2 : idx[0]);
		int ref_col = idx[1] < 0 ? -idx[1] : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] * 2 - idx[1] - 2 : idx[1]);
		return A(ref_row, ref_col);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate_for_row(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_col = idx[1] < 0 ? 0 : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] - 1 : idx[1]);
		return A(idx[0], rep_col);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate_for_col(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_row = idx[0] < 0 ? 0 : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] - 1 : idx[0]);
		return A(rep_row, idx[1]);
	}

	template <typename value_type>
	inline value_type guarded_read_replicate(const array_view<value_type, 2>& A, const concurrency::index<2>& idx) restrict(cpu, amp)
	{
		int rep_row = idx[0] < 0 ? 0 : (idx[0] >= A.get_extent()[0] ? A.get_extent()[0] - 1 : idx[0]);
		int rep_col = idx[1] < 0 ? 0 : (idx[1] >= A.get_extent()[1] ? A.get_extent()[1] - 1 : idx[1]);
		return A(rep_row, rep_col);
	}

	template <typename value_type, int rank>
	inline void guarded_write(const array_view<value_type, rank>& A, const concurrency::index<rank>& idx, const value_type& val) restrict(cpu, amp)
	{
		if(A.get_extent().contains(idx))
			A[idx] = val;
	}

	// Debugging Helpers
	template<typename value_type>
	inline void load_vector(const std::vector<value_type>& srcVec, array_view<value_type, 1> destView)
	{
		assert(srcVec.size() == destView.get_extent()[0]);
		concurrency::copy(srcVec.begin(), srcVec.end(), destView);
	}

	template<typename value_type, int rank>
	inline void save_vector(array_view<const value_type, rank> srcView, std::vector<value_type>& destVec)
	{
		destVec.resize(srcView.get_extent().size());
		concurrency::copy(srcView, &destVec[0]);
	}

	template<typename value_type, int rank>
	inline void save_vector(array_view<value_type, rank> srcView, std::vector<value_type>& destVec)
	{
		destVec.resize(srcView.get_extent().size());
		concurrency::copy(srcView, &destVec[0]);
	}

	template<typename value_type, int rank>
	inline void save_vector(concurrency::array<value_type, rank> srcArray, std::vector<value_type>& destVec)
	{
		destVec.resize(srcArray.get_extent().size());
		concurrency::copy(srcArray, &destVec[0]);
	}

	template<typename T1, typename T2 = T1>
	inline void dump_vector(std::ostream& os, const std::vector<T1>& v, int columns)
	{
		int count = 0;
		for(auto it = v.begin(); it != v.end(); it++)
		{
			os << T2(*it);
			if(++count % columns == 0)
			{
				os << std::endl;
			}
			else
			{
				os << ',';
			}
		}
		if(v.size() % columns != 0)
		{
			os << std::endl;
		}
	}

	// Data Structures
	template<typename T>
	struct is_float
	{
		static const bool value = false;
	};

	template<>
	struct is_float < float >
	{
		static const bool value = true;
	};

	template<typename T = float, unsigned int max_size = 1024>
	struct kernel_wrapper
	{
		kernel_wrapper()
			: rows(0), cols(0), size(0)
		{
			std::memset(data, 0, max_size * sizeof(T));
		}

		kernel_wrapper(int rows_, int cols_, std::initializer_list<T> values)
			: rows(rows_), cols(cols_), size(rows_ * cols_)
		{
			std::memset(data, 0, max_size * sizeof(T));
			std::copy(values.begin(), values.end(), &data[0]);
		}

		template<typename IteratorT>
		kernel_wrapper(IteratorT begin, IteratorT end)
			: rows(1), cols(end - begin), size(end - begin)
		{
			std::memset(data, 0, max_size * sizeof(T));
			std::copy(begin, end, &data[0]);
		}

		explicit kernel_wrapper(const cv::Mat& kernel)
			: rows(kernel.rows), cols(kernel.cols), size(kernel.cols * kernel.rows)
		{
			static_assert(is_float<T>::value, "only support converting to float array");
			assert((unsigned int)size <= max_size);
			assert(kernel.channels() == 1);
			std::memset(data, 0, max_size * sizeof(T));
			if(kernel.type() == CV_32FC1)
			{
				std::memcpy(data, kernel.data, size * sizeof(T));
			}
			else
			{
				cv::Mat cvtKernel;
				kernel.convertTo(cvtKernel, CV_32FC1);
				std::memcpy(data, cvtKernel.data, size * sizeof(T));
			}
		}

		bool is_all_positive() const
		{
			for(int i = 0; i < size; i++)
			{
				if(data[i] <= (T)0)
				{
					return false;
				}
			}
			return true;
		}

		T data[max_size];
		int size;
		int cols;
		int rows;
	};

	namespace detail
	{
		inline void load_cv_mat_8u_c1(accelerator_view& acc_view, array_view<const unsigned int, 1> srcArray, int row_step, array_view<float, 2> destArray)
		{
			static const int tile_size = 32;
			destArray.discard_data();
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int src_byte_index = idx.global[0] * row_step + idx.global[1];
				unsigned int pix4_val = guarded_read(srcArray, concurrency::index<1>(src_byte_index / 4));
				guarded_write(destArray, idx.global, float((pix4_val >> ((src_byte_index % 4) << 3)) & 0xffu));
			});
		}

		inline void load_cv_mat_16u_c1(accelerator_view& acc_view, array_view<const unsigned int, 1> srcArray, int row_step, array_view<float, 2> destArray, float scale = 255.0f / 4095.0f)
		{
			static const int tile_size = 32;
			destArray.discard_data();
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int src_byte_index = idx.global[0] * row_step + idx.global[1] * 2;
				unsigned int pix2_val = guarded_read(srcArray, concurrency::index<1>(src_byte_index / 4));
				guarded_write(destArray, idx.global, float((pix2_val >> ((src_byte_index % 4) << 3)) & 0xffffu) * scale);
			});
		}

		inline void save_cv_mat_8u_c1(accelerator_view& acc_view, array_view<const float, 2> srcArray, int row_step, array_view<unsigned int, 1> destArray)
		{
			using namespace std;
			static const int tile_size = 256;
			destArray.discard_data();
			parallel_for_each(acc_view, destArray.get_extent().tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
			{
				int dest_byte_index = idx.global[0] * 4;
				unsigned int pix0 = (unsigned int)fast_math::roundf(direct3d::clamp(guarded_read(srcArray, concurrency::index<2>(dest_byte_index / row_step, dest_byte_index % row_step)), 0.0f, 255.0f));
				unsigned int pix1 = (unsigned int)fast_math::roundf(direct3d::clamp(guarded_read(srcArray, concurrency::index<2>((dest_byte_index + 1) / row_step, (dest_byte_index + 1) % row_step)), 0.0f, 255.0f));
				unsigned int pix2 = (unsigned int)fast_math::roundf(direct3d::clamp(guarded_read(srcArray, concurrency::index<2>((dest_byte_index + 2) / row_step, (dest_byte_index + 2) % row_step)), 0.0f, 255.0f));
				unsigned int pix3 = (unsigned int)fast_math::roundf(direct3d::clamp(guarded_read(srcArray, concurrency::index<2>((dest_byte_index + 3) / row_step, (dest_byte_index + 3) % row_step)), 0.0f, 255.0f));
				guarded_write(destArray, idx.global, pix0 + (pix1 << 8) + (pix2 << 16) + (pix3 << 24));
			});
		}

		inline void load_cv_mat_8u_c3(accelerator_view& acc_view, array_view<const unsigned int, 1> srcArray, int row_step
			, array_view<float, 2> destChannel1, array_view<float, 2> destChannel2, array_view<float, 2> destChannel3)
		{
			static const int tile_size = 32;
			destChannel1.discard_data();
			destChannel2.discard_data();
			destChannel3.discard_data();
			parallel_for_each(acc_view, destChannel1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int channel1_src_index = idx.global[0] * row_step + idx.global[1] * 3;
				int channel2_src_index = idx.global[0] * row_step + idx.global[1] * 3 + 1;
				int channel3_src_index = idx.global[0] * row_step + idx.global[1] * 3 + 2;
				unsigned int channel1_val = guarded_read(srcArray, concurrency::index<1>(channel1_src_index / 4));
				guarded_write(destChannel1, idx.global, float((channel1_val >> ((channel1_src_index % 4) << 3)) & 0xffu));
				unsigned int channel2_val = guarded_read(srcArray, concurrency::index<1>(channel2_src_index / 4));
				guarded_write(destChannel2, idx.global, float((channel2_val >> ((channel2_src_index % 4) << 3)) & 0xffu));
				unsigned int channel3_val = guarded_read(srcArray, concurrency::index<1>(channel3_src_index / 4));
				guarded_write(destChannel3, idx.global, float((channel3_val >> ((channel3_src_index % 4) << 3)) & 0xffu));
			});
		}

		inline void load_cv_mat_16u_c3(accelerator_view& acc_view, array_view<const unsigned int, 1> srcArray, int row_step
			, array_view<float, 2> destChannel1, array_view<float, 2> destChannel2, array_view<float, 2> destChannel3, float scale = 255.0f / 4095.0f)
		{
			static const int tile_size = 32;
			destChannel1.discard_data();
			destChannel2.discard_data();
			destChannel3.discard_data();
			parallel_for_each(acc_view, destChannel1.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				int channel1_src_index = idx.global[0] * row_step + idx.global[1] * 6;
				int channel2_src_index = idx.global[0] * row_step + idx.global[1] * 6 + 2;
				int channel3_src_index = idx.global[0] * row_step + idx.global[1] * 6 + 4;
				unsigned int channel1_val = guarded_read(srcArray, concurrency::index<1>(channel1_src_index / 4));
				guarded_write(destChannel1, idx.global, float((channel1_val >> ((channel1_src_index % 4) << 3)) & 0xffffu));
				unsigned int channel2_val = guarded_read(srcArray, concurrency::index<1>(channel2_src_index / 4));
				guarded_write(destChannel2, idx.global, float((channel2_val >> ((channel2_src_index % 4) << 3)) & 0xffffu));
				unsigned int channel3_val = guarded_read(srcArray, concurrency::index<1>(channel3_src_index / 4));
				guarded_write(destChannel3, idx.global, float((channel3_val >> ((channel3_src_index % 4) << 3)) & 0xffffu));
			});
		}

		inline unsigned int combine_to_32u(float f1, float f2, float f3, float f4) restrict(amp)
		{
			unsigned int pix1 = (unsigned int)fast_math::roundf(direct3d::clamp(f1, 0.0f, 255.0f));
			unsigned int pix2 = (unsigned int)fast_math::roundf(direct3d::clamp(f2, 0.0f, 255.0f));
			unsigned int pix3 = (unsigned int)fast_math::roundf(direct3d::clamp(f3, 0.0f, 255.0f));
			unsigned int pix4 = (unsigned int)fast_math::roundf(direct3d::clamp(f4, 0.0f, 255.0f));
			return pix1 + (pix2 << 8) + (pix3 << 16) + (pix4 << 24);
		}

		inline void save_cv_mat_8u_c3(accelerator_view& acc_view, array_view<const float, 2> srcChannel1, array_view<const float, 2> srcChannel2, array_view<const float, 2> srcChannel3
			, int row_step, array_view<unsigned int, 1> destArray)
		{
			static const int tile_size = 256;
			destArray.discard_data();
			concurrency::extent<1> ext(DIVUP(row_step * srcChannel1.get_extent()[0], 3));
			parallel_for_each(acc_view, ext.tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
			{
				int dest_byte_index = idx.global[0] * 12;
				int row1 = dest_byte_index / row_step;
				int col1 = (dest_byte_index % row_step) / 3;
				float src1_ch1 = guarded_read(srcChannel1, concurrency::index<2>(row1, col1));
				float src1_ch2 = guarded_read(srcChannel2, concurrency::index<2>(row1, col1));
				float src1_ch3 = guarded_read(srcChannel3, concurrency::index<2>(row1, col1));
				int row2 = (dest_byte_index + 3) / row_step;
				int col2 = ((dest_byte_index + 3) % row_step) / 3;
				float src2_ch1 = guarded_read(srcChannel1, concurrency::index<2>(row2, col2));
				float src2_ch2 = guarded_read(srcChannel2, concurrency::index<2>(row2, col2));
				float src2_ch3 = guarded_read(srcChannel3, concurrency::index<2>(row2, col2));
				int row3 = (dest_byte_index + 6) / row_step;
				int col3 = ((dest_byte_index + 6) % row_step) / 3;
				float src3_ch1 = guarded_read(srcChannel1, concurrency::index<2>(row3, col3));
				float src3_ch2 = guarded_read(srcChannel2, concurrency::index<2>(row3, col3));
				float src3_ch3 = guarded_read(srcChannel3, concurrency::index<2>(row3, col3));
				int row4 = (dest_byte_index + 9) / row_step;
				int col4 = ((dest_byte_index + 9) % row_step) / 3;
				float src4_ch1 = guarded_read(srcChannel1, concurrency::index<2>(row4, col4));
				float src4_ch2 = guarded_read(srcChannel2, concurrency::index<2>(row4, col4));
				float src4_ch3 = guarded_read(srcChannel3, concurrency::index<2>(row4, col4));
				guarded_write(destArray, idx.global * 3, combine_to_32u(src1_ch1, src1_ch2, src1_ch3, src2_ch1));
				guarded_write(destArray, idx.global * 3 + 1, combine_to_32u(src2_ch2, src2_ch3, src3_ch1, src3_ch2));
				guarded_write(destArray, idx.global * 3 + 2, combine_to_32u(src3_ch3, src4_ch1, src4_ch2, src4_ch3));
			});
		}

	}

	class vision_context
	{
	public:
		// Constructor
		vision_context(const accelerator_view& acc_view_, float max_image_mpixels, size_t buffer_size = 16)
			: acc_view(acc_view_), save_load_buf(cvRound(max_image_mpixels * 1024 + 1) * 256, acc_view_)
		{
			// Need a vector container that moves elements when storage is full
			float2d.reserve(buffer_size);
			float1d.reserve(buffer_size);
			int2d.reserve(buffer_size);
			int1d.reserve(buffer_size);
		}


		// Create/Clear operation buffers
		int create_float2d_buf(int rows, int cols, int count = 1)
		{
			size_t index = float2d.size();
			if(index + size_t(count) > float2d.capacity()) throw std::runtime_error("float2d array is full");
			while(count-- > 0)
			{
				float2d.emplace_back(rows, cols, acc_view);
			}
			return int(index);
		}

		int create_float1d_buf(int cols, int count = 1)
		{
			size_t index = float1d.size();
			if(index + size_t(count) > float1d.capacity()) throw std::runtime_error("float1d array is full");
			while(count-- > 0)
			{
				float1d.emplace_back(cols, acc_view);
			}
			return int(index);
		}

		int create_int2d_buf(int rows, int cols, int count = 1)
		{
			size_t index = int2d.size();
			if(index + size_t(count) > int2d.capacity()) throw std::runtime_error("int2d array is full");
			while(count-- > 0)
			{
				int2d.emplace_back(rows, cols, acc_view);
			}
			return int(index);
		}

		int create_int1d_buf(int cols, int count = 1)
		{
			size_t index = int1d.size();
			while(count-- > 0)
			if(index + size_t(count) > int1d.capacity()) throw std::runtime_error("int1d array is full");
			{
				int1d.emplace_back(cols, acc_view);
			}
			return int(index);
		}

		void clear_all_buf()
		{
			float2d.clear();
			float1d.clear();
			int2d.clear();
			int1d.clear();
		}

		// OpenCV Mat upload/download support(8UC1 & 8UC3, 16UC1 & 16UC3[load only])
		bool load_cv_mat(const cv::Mat& srcMat, array_view<float, 2> destView)
		{
			// check Mat type and size
			if ((srcMat.type() != CV_8UC1 && srcMat.type() != CV_16UC1) || srcMat.rows != destView.get_extent()[0] || srcMat.cols != destView.get_extent()[1])
			{
				return false;
			}
			// clone if source Mat is not continuous
			cv::Mat continousMat;
			if(!srcMat.isContinuous())
			{
				srcMat.copyTo(continousMat);
			}
			else
			{
				continousMat = srcMat;
			}
            // check required buffer size
			int source_dword_count = (continousMat.rows * int(continousMat.step[0]) + 3) / 4;
			if (source_dword_count > save_load_buf.get_extent()[0])
			{
				std::swap(save_load_buf, concurrency::array<unsigned int, 1>(ROUNDUP(source_dword_count, 64), acc_view));
			}
			// copy data to GPU
            concurrency::copy(continousMat.ptr<unsigned int>(), continousMat.ptr<unsigned int>() + source_dword_count, save_load_buf.section(0, source_dword_count));
            // convert data
			if (srcMat.type() == CV_8UC1)
			{
				detail::load_cv_mat_8u_c1(acc_view, save_load_buf, continousMat.step[0], destView);
			}
			else // CV_16UC1
			{
				detail::load_cv_mat_16u_c1(acc_view, save_load_buf, continousMat.step[0], destView);
			}
			return true;
		}

		bool save_cv_mat(array_view<const float, 2> srcView, cv::Mat& destMat)
		{
			int row_step = destMat.step[0];
			// check destMat type and size
			if(destMat.type() != CV_8UC1 || destMat.rows != srcView.get_extent()[0] || destMat.cols != srcView.get_extent()[1] || !destMat.isContinuous())
			{
				// recreate
				destMat = cv::Mat(srcView.get_extent()[0], srcView.get_extent()[1], CV_8UC1);
				row_step = destMat.step[0];
			}
			// convert data
			int source_dword_count = (srcView.get_extent()[0] * row_step + 3) / 4;
			if (source_dword_count > save_load_buf.get_extent()[0])
			{
				std::swap(save_load_buf, concurrency::array<unsigned int, 1>(ROUNDUP(source_dword_count, 64), acc_view));
			}
			array_view<unsigned int> save_load_buf_section = save_load_buf.section(0, source_dword_count);
			detail::save_cv_mat_8u_c1(acc_view, srcView, row_step, save_load_buf_section);
			// download to destMat
			concurrency::copy(save_load_buf_section, (unsigned int*)destMat.data);
			return true;
		}

		bool load_cv_mat(const cv::Mat& srcMat, array_view<float, 2> destChannel1, array_view<float, 2> destChannel2, array_view<float, 2> destChannel3)
		{
			// check Mat type and size
			if ((srcMat.type() != CV_8UC3 && srcMat.type() != CV_16UC3) || srcMat.rows != destChannel1.get_extent()[0] || srcMat.cols != destChannel1.get_extent()[1])
			{
				return false;
			}
			// clone if source Mat is not continuous
			cv::Mat continousMat;
			if(!srcMat.isContinuous())
			{
				srcMat.copyTo(continousMat);
			}
			else
			{
				continousMat = srcMat;
			}
			// check required buffer size
			int source_dword_count = (continousMat.rows * int(continousMat.step[0]) + 3) / 4;
			if(source_dword_count > save_load_buf.get_extent()[0])
			{
				std::swap(save_load_buf, concurrency::array<unsigned int, 1>(ROUNDUP(source_dword_count, 64), acc_view));
			}
			// copy data to GPU
			concurrency::copy_async(continousMat.ptr<unsigned int>(), continousMat.ptr<unsigned int>() + source_dword_count, save_load_buf);
			// convert data
			if (srcMat.type() == CV_8UC3)
			{
				detail::load_cv_mat_8u_c3(acc_view, save_load_buf, continousMat.step[0], destChannel1, destChannel2, destChannel3);
			}
			else
			{
				detail::load_cv_mat_16u_c3(acc_view, save_load_buf, continousMat.step[0], destChannel1, destChannel2, destChannel3);
			}
			return true;
		}

		bool save_cv_mat(array_view<const float, 2> srcChannel1, array_view<const float, 2> srcChannel2, array_view<const float, 2> srcChannel3, cv::Mat& destMat)
		{
			int row_step = destMat.step[0];
			// check destMat type and size
			if(destMat.type() != CV_8UC3 || destMat.rows != srcChannel1.get_extent()[0] || destMat.cols != srcChannel1.get_extent()[1] || !destMat.isContinuous())
			{
				// recreate
				destMat = cv::Mat(srcChannel1.get_extent()[0], srcChannel1.get_extent()[1], CV_8UC3);
				row_step = destMat.step[0];
			}
			// convert data
			int source_dword_count = (srcChannel1.get_extent()[0] * row_step + 3) / 4;
			if (source_dword_count > save_load_buf.get_extent()[0])
			{
				std::swap(save_load_buf, concurrency::array<unsigned int, 1>(ROUNDUP(source_dword_count, 64), acc_view));
			}
			array_view<unsigned int> save_load_buf_section = save_load_buf.section(0, source_dword_count);
			detail::save_cv_mat_8u_c3(acc_view, srcChannel1, srcChannel2, srcChannel3, row_step, save_load_buf_section);
			// download to destMat
			concurrency::copy(save_load_buf_section, (unsigned int*)destMat.data);
			return true;
		}

	public:
		accelerator_view acc_view;
		concurrency::array<unsigned int, 1> save_load_buf;
		std::vector<concurrency::array<float, 2>> float2d;
		std::vector<concurrency::array<float, 1>> float1d;
		std::vector<concurrency::array<int, 2>> int2d;
		std::vector<concurrency::array<int, 1>> int1d;
	};

}
