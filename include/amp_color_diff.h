#pragma once

#include "amp_core.h"

namespace amp
{
	inline float ciede1976_delta_e(float_3 color1, float_3 color2) restrict(amp)
	{
		float_3 diff = color1 - color2;
		return direct3d::mad(diff.x, diff.x, direct3d::mad(diff.y, diff.y, diff.z * diff.z));
	}

	inline float ciede2000_delta_e(float_3 color1, float_3 color2) restrict(amp)
	{
		using fast_math::sqrtf;
		using fast_math::powf;
		using fast_math::atan2f;
		using fast_math::fabsf;
		using fast_math::sinf;
		using fast_math::cosf;
		using fast_math::expf;
		float dLq = color2.x - color1.x;
		float avgL = (color1.x + color2.x) / 2.0f;
		float c1 = sqrtf(color1.y * color1.y + color1.z * color1.z);
		float c2 = sqrtf(color2.y * color2.y + color2.z * color2.z);
		float avgC = (c1 + c2) / 2.0f;
		float avgcPf = (1.0f - sqrtf(powf(avgC, 7.0f) / (powf(avgC, 7.0f) + powf(25.0f, 7.0f)))) / 2.0f;
		float a1q = color1.y + color1.y * avgcPf;
		float a2q = color2.y + color2.y * avgcPf;
		float c1q = sqrtf(a1q * a1q + color1.z * color1.z);
		float c2q = sqrtf(a2q * a2q + color2.z * color2.z);
		float avgCq = (c1q + c2q) / 2.0f;
		float dCq = c2q - c1q;
		float h1q = (a1q != 0.0f || color1.z != 0.0f) ? atan2f(color1.z, a1q) : 0.0f;
		if (h1q < 0.0f) h1q += float(2.0 * CV_PI);
		float h2q = (a2q != 0.0f || color2.z != 0.0f) ? atan2f(color2.z, a2q) : 0.0f;
		if (h2q < 0.0f) h2q += float(2.0 * CV_PI);
		float dhq = fabsf(h1q - h2q) <= CV_PI ? dhq = h2q - h1q : (h2q > h1q ? h2q - h1q - float(CV_PI * 2.0) : h2q - h1q + float(CV_PI * 2.0));
		float dHq = 2.0f * sqrtf(c1q * c2q) * sinf(dhq / 2.0f);
		float avgHq = c1q == 0.0f || c2q == 0.0f ? avgHq = h2q + h1q : (fabsf(h1q - h2q) <= CV_PI ? (h2q + h1q) / 2.0f : (h2q + h1q + 2.0f * float(CV_PI)) / 2.0f);
		float T = 1.0f - 0.17f * cosf(avgHq - float(CV_PI / 6.0)) + 0.24f * cosf(2.0f * avgHq) + 0.32f * cosf(3.0f * avgHq + float(CV_PI / 30.0)) - 0.20f * cosf(4.0f * avgHq - float(CV_PI * 63.0 / 180.0));
		float SL = 1.0f + 0.015f * (avgL - 50.0f) * (avgL - 50.0f) / sqrtf(20.0f + (avgL - 50.0f) * (avgL - 50.0f));
		float SC = 1.0f + 0.045f * avgCq;
		float SH = 1.0f + 0.015f * avgCq * T;
		float RT = -2.0f * sqrtf(powf(avgCq, 7.0f) / (powf(avgCq, 7.0f) + powf(25.0f, 7.0f))) * sinf(float(CV_PI / 3.0) * expf(-powf((avgHq * float(180.0 / CV_PI) - 275.0f) / 25.0f, 2.0f)));
		return sqrtf(powf(dLq / SL, 2.0f) + powf(dCq / SC, 2.0f) + powf(dHq / SH, 2.0f) + RT * dCq / SC * dHq / SH);
	}
}
