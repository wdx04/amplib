#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include "amp_core.h"

namespace amp
{
	inline void convert_bgr_to_grayscale_32f(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_array)
	{
		static const int tile_size = 32;
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_array.get_extent().contains(idx.global))
			{
				float src_b_value = src_channel1(idx.global);
				float src_g_value = src_channel2(idx.global);
				float src_r_value = src_channel3(idx.global);
				dest_array(idx.global) = direct3d::mad(src_b_value, 0.114f, direct3d::mad(src_g_value, 0.587f, src_r_value * 0.299f));
			}
		});
	}

	inline void convert_bayer_to_grayscale_32f(accelerator_view& acc_view, array_view<const float, 2> src_array
		, array_view<float, 2> dest_array, std::pair<int, int> first_red)
	{
		static const int tile_size = 32;
		dest_array.discard_data();
		parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if(dest_array.get_extent().contains(gidx))
			{
				int_2 firstRed(first_red.first, first_red.second);

				// Gets information about work-item (which pixel to process)
				// Add offset for first red
				int_4 center;
				center.xy = int_2(gidx[1], gidx[0]);
				center.zw = center.xy + firstRed;

				// Gets information about work size (dimensions of mosaic)
				int width = dest_array.get_extent()[1];
				int height = dest_array.get_extent()[0];

				int_4 xCoord = center.x + int_4(-2, -1, 1, 2);
				int_4 yCoord = center.y + int_4(-2, -1, 1, 2);

				float C = src_array(gidx); // ( 0, 0)

				const float_4 kC = float_4(4.0f, 6.0f, 5.0f, 5.0f) / 8.0f;

				// Determine which of four types of pixels we are on.
				float_2 alternate(float(center.z % 2), float(center.w % 2));

				float_4 Dvec(guarded_read(src_array, concurrency::index<2>(yCoord.y, xCoord.y)),  // (-1,-1)
					guarded_read(src_array, concurrency::index<2>(yCoord.z, xCoord.y)),  // (-1, 1)
					guarded_read(src_array, concurrency::index<2>(yCoord.y, xCoord.z)),  // ( 1,-1)
					guarded_read(src_array, concurrency::index<2>(yCoord.z, xCoord.z))); // ( 1, 1)

				float_4 PATTERN = kC * C;

				// Equivalent to:  
				// float D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;
				Dvec.xy += Dvec.zw;
				Dvec.x += Dvec.y;

				float_4 value = float_4(guarded_read(src_array, concurrency::index<2>(yCoord.x, center.x)),  // ( 0,-2) A0
					guarded_read(src_array, concurrency::index<2>(yCoord.y, center.x)),  // ( 0,-1) B0
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.x)),  // (-2, 0) E0
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.y))); // (-1, 0) F0

				float_4 temp = float_4(guarded_read(src_array, concurrency::index<2>(yCoord.w, center.x)),  // ( 0, 2) A1
					guarded_read(src_array, concurrency::index<2>(yCoord.z, center.x)),  // ( 0, 1) B1
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.w)),  // ( 2, 0) E1
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.z))); // ( 1, 0) F1

				const float_4 kA = float_4(-1.0f, -1.5f, 0.5f, -1.0f) / 8.0f;
				const float_4 kB = float_4(2.0f, 0.0f, 0.0f, 4.0f) / 8.0f;
				const float_4 kD = float_4(0.0f, 2.0f, -1.0f, -1.0f) / 8.0f;

				// Conserve constant registers and take advantage of free swizzle on load
#define kE kA.xywz
#define kF kB.xywz
				value += temp;  // (A0 + A1), (B0 + B1), (E0 + E1), (F0 + F1)

				// There are five filter patterns (identity, cross, checker,
				// theta, phi). Precompute the terms from all of them and then
				// use swizzles to assign to color channels.
				//
				// Channel Matches
				// x cross (e.g., EE G)
				// y checker (e.g., EE B)
				// z theta (e.g., EO R)
				// w phi (e.g., EO B)
#define A value.x  // A0 + A1
#define B value.y  // B0 + B1
#define D Dvec.x    // D0 + D1 + D2 + D3
#define E value.z  // E0 + E1
#define F value.w  // F0 + F1

				// Avoid zero elements. On a scalar processor this saves two MADDs and it has no
				// effect on a vector processor.
				// PATTERN.yzw += (kD.yz * D).xyy;  <- invalid in OpenCL
				float_2 kDtemp = kD.yz * D;
				PATTERN.yz += kDtemp.xy;
				PATTERN.w += kDtemp.y;

				// PATTERN += (kA.xyz * A).xyzx + (kE.xyw * E).xyxz;  <- invalid in OpenCL
				float_3 kAtemp = kA.xyz * A;
				float_3 kEtemp = kE.xyw * E;
				PATTERN.xyz += kAtemp;
				PATTERN.w += kAtemp.x;
				PATTERN.xy += kEtemp.xy;
				PATTERN.zw += kEtemp.xz;
				PATTERN.xw += kB.xw * B;
				PATTERN.xz += kF.xz * F;

				// in RGB sequence
				float_3 pixelColor = (alternate.y == 0.0f) ?
					((alternate.x == 0.0f) ?
					float_3(C, PATTERN.x, PATTERN.y) :
					float_3(PATTERN.z, C, PATTERN.w)) :
					((alternate.x == 0.0f) ?
					float_3(PATTERN.w, C, PATTERN.z) :
					float_3(PATTERN.y, PATTERN.x, C));

				// output needed in BGR sequence
				dest_array(idx.global) = direct3d::clamp(direct3d::mad(pixelColor.z, 0.114f, direct3d::mad(pixelColor.y, 0.587f, pixelColor.x * 0.299f)), 0.0f, 255.0f);

#undef A
#undef B
#undef C
#undef E
#undef F
#undef kE
#undef kF
			}
		});
	}

	inline void convert_bayer_to_bgr_32f(accelerator_view& acc_view, array_view<const float, 2> src_array
		, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2, array_view<float, 2> dest_channel3, std::pair<int, int> first_red)
	{
		static const int tile_size = 32;
		dest_channel1.discard_data();
		dest_channel2.discard_data();
		dest_channel3.discard_data();
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			concurrency::index<2> gidx = idx.global;
			if(dest_channel1.get_extent().contains(gidx))
			{
				int_2 firstRed(first_red.first, first_red.second);

				// Gets information about work-item (which pixel to process)
				// Add offset for first red
				int_4 center;
				center.xy = int_2(gidx[1], gidx[0]);
				center.zw = center.xy + firstRed;

				// Gets information about work size (dimensions of mosaic)
				int width = dest_channel1.get_extent()[1];
				int height = dest_channel1.get_extent()[0];

				int_4 xCoord = center.x + int_4(-2, -1, 1, 2);
				int_4 yCoord = center.y + int_4(-2, -1, 1, 2);

				float C = src_array(gidx); // ( 0, 0)

				const float_4 kC = float_4(4.0f, 6.0f, 5.0f, 5.0f) / 8.0f;

				// Determine which of four types of pixels we are on.
				float_2 alternate(float(center.z % 2), float(center.w % 2));

				float_4 Dvec(guarded_read(src_array, concurrency::index<2>(yCoord.y, xCoord.y)),  // (-1,-1)
					guarded_read(src_array, concurrency::index<2>(yCoord.z, xCoord.y)),  // (-1, 1)
					guarded_read(src_array, concurrency::index<2>(yCoord.y, xCoord.z)),  // ( 1,-1)
					guarded_read(src_array, concurrency::index<2>(yCoord.z, xCoord.z))); // ( 1, 1)

				float_4 PATTERN = kC * C;

				// Equivalent to:  
				// float D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;
				Dvec.xy += Dvec.zw;
				Dvec.x += Dvec.y;

				float_4 value = float_4(guarded_read(src_array, concurrency::index<2>(yCoord.x, center.x)),  // ( 0,-2) A0
					guarded_read(src_array, concurrency::index<2>(yCoord.y, center.x)),  // ( 0,-1) B0
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.x)),  // (-2, 0) E0
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.y))); // (-1, 0) F0

				float_4 temp = float_4(guarded_read(src_array, concurrency::index<2>(yCoord.w, center.x)),  // ( 0, 2) A1
					guarded_read(src_array, concurrency::index<2>(yCoord.z, center.x)),  // ( 0, 1) B1
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.w)),  // ( 2, 0) E1
					guarded_read(src_array, concurrency::index<2>(center.y, xCoord.z))); // ( 1, 0) F1

				const float_4 kA = float_4(-1.0f, -1.5f, 0.5f, -1.0f) / 8.0f;
				const float_4 kB = float_4(2.0f, 0.0f, 0.0f, 4.0f) / 8.0f;
				const float_4 kD = float_4(0.0f, 2.0f, -1.0f, -1.0f) / 8.0f;

				// Conserve constant registers and take advantage of free swizzle on load
#define kE kA.xywz
#define kF kB.xywz
				value += temp;  // (A0 + A1), (B0 + B1), (E0 + E1), (F0 + F1)

				// There are five filter patterns (identity, cross, checker,
				// theta, phi). Precompute the terms from all of them and then
				// use swizzles to assign to color channels.
				//
				// Channel Matches
				// x cross (e.g., EE G)
				// y checker (e.g., EE B)
				// z theta (e.g., EO R)
				// w phi (e.g., EO B)
#define A value.x  // A0 + A1
#define B value.y  // B0 + B1
#define D Dvec.x    // D0 + D1 + D2 + D3
#define E value.z  // E0 + E1
#define F value.w  // F0 + F1

				// Avoid zero elements. On a scalar processor this saves two MADDs and it has no
				// effect on a vector processor.
				// PATTERN.yzw += (kD.yz * D).xyy;  <- invalid in OpenCL
				float_2 kDtemp = kD.yz * D;
				PATTERN.yz += kDtemp.xy;
				PATTERN.w += kDtemp.y;

				// PATTERN += (kA.xyz * A).xyzx + (kE.xyw * E).xyxz;  <- invalid in OpenCL
				float_3 kAtemp = kA.xyz * A;
				float_3 kEtemp = kE.xyw * E;
				PATTERN.xyz += kAtemp;
				PATTERN.w += kAtemp.x;
				PATTERN.xy += kEtemp.xy;
				PATTERN.zw += kEtemp.xz;
				PATTERN.xw += kB.xw * B;
				PATTERN.xz += kF.xz * F;

				// in RGB sequence
				float_3 pixelColor = (alternate.y == 0.0f) ?
					((alternate.x == 0.0f) ?
					float_3(C, PATTERN.x, PATTERN.y) :
					float_3(PATTERN.z, C, PATTERN.w)) :
					((alternate.x == 0.0f) ?
					float_3(PATTERN.w, C, PATTERN.z) :
					float_3(PATTERN.y, PATTERN.x, C));

				// output needed in BGR sequence
				dest_channel1(gidx) = direct3d::clamp(pixelColor.z, 0.0f, 255.0f);
				dest_channel2(gidx) = direct3d::clamp(pixelColor.y, 0.0f, 255.0f);
				dest_channel3(gidx) = direct3d::clamp(pixelColor.x, 0.0f, 255.0f);

#undef A
#undef B
#undef C
#undef E
#undef F
#undef kE
#undef kF
			}
		});
	}

	inline void convert_bgr_to_hsv_32f(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3)
	{
		static const int tile_size = 32;
		dest_channel1.discard_data();
		dest_channel2.discard_data();
		dest_channel3.discard_data();
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_channel1.get_extent().contains(idx.global))
			{
				float b_value = src_channel1(idx.global);
				float g_value = src_channel2(idx.global);
				float r_value = src_channel3(idx.global);
				float h_value, s_value, v_value;
				float vmin, diff;
				v_value = vmin = r_value;
				if(v_value < g_value) v_value = g_value;
				if(v_value < b_value) v_value = b_value;
				if(vmin > g_value) vmin = g_value;
				if(vmin > b_value) vmin = b_value;
				diff = v_value - vmin;
				s_value = diff / (float)(fast_math::fabs(v_value) + FLT_EPSILON);
				diff = (float)(60.f / (diff + FLT_EPSILON));
				if(v_value == r_value)
					h_value = (g_value - b_value)*diff;
				else if(v_value == g_value)
					h_value = direct3d::mad(b_value - r_value, diff, 120.f);
				else
					h_value = direct3d::mad(r_value - g_value, diff, 240.f);
				if(h_value < 0)
					h_value += 360.f;
				dest_channel1(idx.global) = h_value;
				dest_channel2(idx.global) = s_value;
				dest_channel3(idx.global) = v_value;
			}
		});
	}

	inline void convert_hsv_to_bgr_32f(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3)
	{
		static const int tile_size = 32;
		kernel_wrapper<int, 18U> sector_data(6, 3, { 1, 3, 0, 1, 0, 2, 3, 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 0 });
		dest_channel1.discard_data();
		dest_channel2.discard_data();
		dest_channel3.discard_data();
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_channel1.get_extent().contains(idx.global))
			{
				float h_value = src_channel1(idx.global) / 60.0f;
				float s_value = src_channel2(idx.global);
				float v_value = src_channel3(idx.global);
				float b_value, g_value, r_value;
				if(s_value != 0.0f)
				{
					float tab[4];
					int sector;
					if(h_value < 0)
						do h_value += 6; while(h_value < 0);
					else if(h_value >= 6)
						do h_value -= 6; while(h_value >= 6);
					sector = int(fast_math::floorf(h_value)); // was rtn
					h_value -= sector;
					if((unsigned)sector >= 6u)
					{
						sector = 0;
						h_value = 0.0f;
					}
					tab[0] = v_value;
					tab[1] = v_value*(1.0f - s_value);
					tab[2] = v_value*(1.0f - s_value * h_value);
					tab[3] = v_value*(1.0f - s_value * (1.0f - h_value));
					b_value = tab[sector_data.data[sector * 3]];
					g_value = tab[sector_data.data[sector * 3 + 1]];
					r_value = tab[sector_data.data[sector * 3 + 2]];
				}
				else
				{
					b_value = g_value = r_value = v_value;
				}
				dest_channel1(idx.global) = b_value;
				dest_channel2(idx.global) = g_value;
				dest_channel3(idx.global) = r_value;
			}
		});
	}

	// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
	template<typename _Tp> static inline void splineBuild(const _Tp* f, int n, _Tp* tab)
	{
		_Tp cn = 0;
		int i;
		tab[0] = tab[1] = (_Tp)0;
		for(i = 1; i < n - 1; i++)
		{
			_Tp t = 3 * (f[i + 1] - 2 * f[i] + f[i - 1]);
			_Tp l = 1 / (4 - tab[(i - 1) * 4]);
			tab[i * 4] = l; tab[i * 4 + 1] = (t - tab[(i - 1) * 4 + 1])*l;
		}
		for(i = n - 1; i >= 0; i--)
		{
			_Tp c = tab[i * 4 + 1] - tab[i * 4] * cn;
			_Tp b = f[i + 1] - f[i] - (cn + c * 2)*(_Tp)0.3333333333333333;
			_Tp d = (cn - c)*(_Tp)0.3333333333333333;
			tab[i * 4] = f[i]; tab[i * 4 + 1] = b;
			tab[i * 4 + 2] = c; tab[i * 4 + 3] = d;
			cn = c;
		}
	}

	// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
	template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
	{
		// don't touch this function without urgent need - some versions of gcc fail to inline it correctly
		int ix = std::min(std::max(int(x), 0), n - 1);
		x -= ix;
		tab += ix * 4;
		return ((tab[3] * x + tab[2])*x + tab[1])*x + tab[0];
	}

	inline float splineInterpolate(float x, float *tab, int n) restrict(amp)
	{
		int ix = direct3d::clamp(int(fast_math::floorf(x)), 0, n - 1);
		x -= ix;
		tab += ix << 2;
		return direct3d::mad(direct3d::mad(direct3d::mad(tab[3], x, tab[2]), x, tab[1]), x, tab[0]);
	}

	static const float D65[] = { 0.950456f, 1.f, 1.088754f };
	enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
	static float LabCbrtTab[LAB_CBRT_TAB_SIZE * 4];
	static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE / 1.5f;
	static float sRGBGammaTab[GAMMA_TAB_SIZE * 4], sRGBInvGammaTab[GAMMA_TAB_SIZE * 4];
	static const float GammaTabScale = (float)GAMMA_TAB_SIZE;
	static inline void initLabTabs()
	{
		static bool initialized = false;
		if(!initialized)
		{
			float f[LAB_CBRT_TAB_SIZE + 1], g[GAMMA_TAB_SIZE + 1], ig[GAMMA_TAB_SIZE + 1], scale = 1.f / LabCbrtTabScale;
			int i;
			for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
			{
				float x = i*scale;
				f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);
			}
			splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

			scale = 1.f / GammaTabScale;
			for(i = 0; i <= GAMMA_TAB_SIZE; i++)
			{
				float x = i*scale;
				g[i] = x <= 0.04045f ? x*(1.f / 12.92f) : (float)std::pow((double)(x + 0.055)*(1. / 1.055), 2.4);
				ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1. / 2.4) - 0.055);
			}
			splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
			splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

			initialized = true;
		}
	}

	static const float sRGB2XYZ_D65[] =
	{
		0.412453f, 0.357580f, 0.180423f,
		0.212671f, 0.715160f, 0.072169f,
		0.019334f, 0.119193f, 0.950227f
	};

	static const float XYZ2sRGB_D65[] =
	{
		3.240479f, -1.53715f, -0.498535f,
		-0.969256f, 1.875991f, 0.041556f,
		0.055648f, -0.204043f, 1.057311f
	};

	inline void convert_bgr_to_lab_32f(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3)
	{
		initLabTabs();
		concurrency::array<float, 1> deviceGammaTab(GAMMA_TAB_SIZE * 4, acc_view);
		concurrency::copy(sRGBGammaTab, deviceGammaTab);
		kernel_wrapper<float, 9U> coeffs;
		const float * const _coeffs = sRGB2XYZ_D65, *const _whitept = D65;
		float scale[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

		for(int i = 0; i < 3; i++)
		{
			int j = i * 3;
			coeffs.data[j] = _coeffs[j] * scale[i];
			coeffs.data[j + 1] = _coeffs[j + 1] * scale[i];
			coeffs.data[j + 2] = _coeffs[j + 2] * scale[i];
		}

		float d = 1.f / (_whitept[0] + _whitept[1] * 15 + _whitept[2] * 3);
		float un = 13 * 4 * _whitept[0] * d;
		float vn = 13 * 9 * _whitept[1] * d;
		float _1_3 = 1.0f / 3.0f;
		float _a = 16.0f / 116.0f;
		float gamma_tab_scale = float(GAMMA_TAB_SIZE);
		static const int tile_size = 32;
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=, &deviceGammaTab](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_channel1.get_extent().contains(idx.global))
			{
				float b_value = direct3d::clamp(src_channel1(idx.global) / 255.0f, 0.0f, 1.0f);
				float g_value = direct3d::clamp(src_channel2(idx.global) / 255.0f, 0.0f, 1.0f);
				float r_value = direct3d::clamp(src_channel3(idx.global) / 255.0f, 0.0f, 1.0f);
				r_value = splineInterpolate(r_value * gamma_tab_scale, &deviceGammaTab[0], GAMMA_TAB_SIZE);
				g_value = splineInterpolate(g_value * gamma_tab_scale, &deviceGammaTab[0], GAMMA_TAB_SIZE);
				b_value = splineInterpolate(b_value * gamma_tab_scale, &deviceGammaTab[0], GAMMA_TAB_SIZE);
				float X = direct3d::mad(r_value, coeffs.data[0], direct3d::mad(g_value, coeffs.data[1], b_value * coeffs.data[2]));
				float Y = direct3d::mad(r_value, coeffs.data[3], direct3d::mad(g_value, coeffs.data[4], b_value * coeffs.data[5]));
				float Z = direct3d::mad(r_value, coeffs.data[6], direct3d::mad(g_value, coeffs.data[7], b_value * coeffs.data[8]));
				float FX = X > 0.008856f ? fast_math::powf(X, _1_3) : direct3d::mad(7.787f, X, _a);
				float FY = Y > 0.008856f ? fast_math::powf(Y, _1_3) : direct3d::mad(7.787f, Y, _a);
				float FZ = Z > 0.008856f ? fast_math::powf(Z, _1_3) : direct3d::mad(7.787f, Z, _a);
				float L = Y > 0.008856f ? direct3d::mad(116.f, FY, -16.0f) : (903.3f * Y);
				float a = 500.f * (FX - FY);
				float b = 200.f * (FY - FZ);
				dest_channel1(idx.global) = L;
				dest_channel2(idx.global) = a;
				dest_channel3(idx.global) = b;
			}
		});
	}

	inline void convert_lab_to_bgr_32f(accelerator_view& acc_view, array_view<const float, 2> src_channel1, array_view<const float, 2> src_channel2
		, array_view<const float, 2> src_channel3, array_view<float, 2> dest_channel1, array_view<float, 2> dest_channel2
		, array_view<float, 2> dest_channel3)
	{
		initLabTabs();
		concurrency::array<float, 1> deviceInvGammaTab(GAMMA_TAB_SIZE * 4, acc_view);
		concurrency::copy(sRGBInvGammaTab, deviceInvGammaTab);
		kernel_wrapper<float, 9U> coeffs;
		const float * const _coeffs = XYZ2sRGB_D65, *const _whitept = D65;
		for(int i = 0; i < 3; i++)
		{
			coeffs.data[i] = _coeffs[i] * _whitept[i];
			coeffs.data[i + 3] = _coeffs[i + 3] * _whitept[i];
			coeffs.data[i + 6] = _coeffs[i + 6] * _whitept[i];
		}
		float d = 1.0f / (_whitept[0] + _whitept[1] * 15 + _whitept[2] * 3);
		float un = 4 * _whitept[0] * d;
		float vn = 9 * _whitept[1] * d;
		float lThresh = 0.008856f * 903.3f;
		float fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
		float gamma_tab_scale = float(GAMMA_TAB_SIZE);
		static const int tile_size = 32;
		parallel_for_each(acc_view, dest_channel1.get_extent().tile<tile_size, tile_size>().pad(), [=, &deviceInvGammaTab](concurrency::tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			if(dest_channel1.get_extent().contains(idx.global))
			{
				float li = src_channel1(idx.global);
				float ai = src_channel2(idx.global);
				float bi = src_channel3(idx.global);
				float y, fy;
				if(li <= lThresh)
				{
					y = li / 903.3f;
					fy = direct3d::mad(7.787f, y, 16.0f / 116.0f);
				}
				else
				{
					fy = (li + 16.0f) / 116.0f;
					y = fy * fy * fy;
				}
				float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
				if(fxz[0] <= fThresh)
					fxz[0] = (fxz[0] - 16.0f / 116.0f) / 7.787f;
				else
					fxz[0] = fxz[0] * fxz[0] * fxz[0];
				if(fxz[1] <= fThresh)
					fxz[1] = (fxz[1] - 16.0f / 116.0f) / 7.787f;
				else
					fxz[1] = fxz[1] * fxz[1] * fxz[1];
				float x = fxz[0], z = fxz[1];
				float ro = direct3d::clamp(direct3d::mad(coeffs.data[0], x, direct3d::mad(coeffs.data[1], y, coeffs.data[2] * z)), 0.0f, 1.0f);
				float go = direct3d::clamp(direct3d::mad(coeffs.data[3], x, direct3d::mad(coeffs.data[4], y, coeffs.data[5] * z)), 0.0f, 1.0f);
				float bo = direct3d::clamp(direct3d::mad(coeffs.data[6], x, direct3d::mad(coeffs.data[7], y, coeffs.data[8] * z)), 0.0f, 1.0f);
				dest_channel1(idx.global) = splineInterpolate(bo * gamma_tab_scale, &deviceInvGammaTab[0], GAMMA_TAB_SIZE) * 255.0f;
				dest_channel2(idx.global) = splineInterpolate(go * gamma_tab_scale, &deviceInvGammaTab[0], GAMMA_TAB_SIZE) * 255.0f;
				dest_channel3(idx.global) = splineInterpolate(ro * gamma_tab_scale, &deviceInvGammaTab[0], GAMMA_TAB_SIZE) * 255.0f;
			}
		});
	}
}
