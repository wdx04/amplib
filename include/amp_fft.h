#pragma once
//--------------------------------------------------------------------------------------
// File: amp_fft.h
//
// Header file for the C++ AMP wrapper over the Direct3D FFT API's.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <d3d11.h>
#include <wrl\client.h>
#include <d3dcsx.h>
#pragma comment(lib, "d3dcsx")
#include "amp_core.h"
#include "amp_copy_make_border.h"

namespace amp
{
	//--------------------------------------------------------------------------------------
	// This exception type should be expected from all public methods in this header, 
	// except for destructors.
	//--------------------------------------------------------------------------------------
	class fft_exception : public std::exception
	{
	public:
		explicit fft_exception(HRESULT error_code) throw()
			: err_code(error_code) {}

		fft_exception(const char *const& msg, HRESULT error_code) throw()
			: err_msg(msg), err_code(error_code) {}

		fft_exception(const std::string& msg, HRESULT error_code) throw()
			: err_msg(msg), err_code(error_code) {}

		fft_exception(const fft_exception &other) throw()
			: std::exception(other), err_msg(other.err_msg), err_code(other.err_code) {}

		virtual ~fft_exception() throw() {}

		HRESULT get_error_code() const throw()
		{
			return err_code;
		}

		virtual const char *what() const throw()
		{
			return  err_msg.data();
		}

	private:
		fft_exception &operator=(const fft_exception &);
		std::string err_msg;
		HRESULT err_code;
	};

	//--------------------------------------------------------------------------------------
	// Implementation details, amp_fft.cpp contains the implementation for the functions
	// declared in this namespace. 
	//--------------------------------------------------------------------------------------
	namespace _details
	{
		template <typename _Type>
		struct dx_fft_type_helper
		{
			static const bool is_type_supported = false;
		};

		template <>
		struct dx_fft_type_helper < float >
		{
			static const bool is_type_supported = true;
			typedef float precision_type;
			static const D3DX11_FFT_DATA_TYPE dx_type = D3DX11_FFT_DATA_TYPE_REAL;
		};

		template <>
		struct dx_fft_type_helper < std::complex<float> >
		{
			static const bool is_type_supported = true;
			typedef float precision_type;
			static const D3DX11_FFT_DATA_TYPE dx_type = D3DX11_FFT_DATA_TYPE_COMPLEX;
		};

		class fft_base
		{
		public:
			void set_forward_scale(float scale)
			{
				if(scale == 0.0f)
					throw fft_exception("Invalid scale value in set_forward_scale", E_INVALIDARG);

				HRESULT hr = _M_pFFT->SetForwardScale(scale);
				if(FAILED(hr)) throw fft_exception("set_forward_scale failed", hr);
			}

			float get_forward_scale() const
			{
				return _M_pFFT->GetForwardScale();
			}

			void set_inverse_scale(float scale)
			{
				if(scale == 0.0f)
					throw fft_exception("Invalid scale value in set_inverse_scale", E_INVALIDARG);

				HRESULT hr = _M_pFFT->SetInverseScale(scale);
				if(FAILED(hr)) throw fft_exception("set_inverse_scale failed", hr);
			}

			float get_inverse_scale() const
			{
				return _M_pFFT->GetInverseScale();
			}

		protected:
			fft_base(D3DX11_FFT_DATA_TYPE _Dx_type, int _Dim, const int* _Transform_extent, const concurrency::accelerator_view& _Av, float _Forward_scale, float _Inverse_scale)
			{
				HRESULT hr = S_OK;

				// Get a device context
				IUnknown *u = concurrency::direct3d::get_device(_Av);
				_ASSERTE(_M_pDevice.Get() == nullptr);
				u->QueryInterface(__uuidof(ID3D11Device), reinterpret_cast<void**>(_M_pDevice.GetAddressOf()));
				_ASSERTE(_M_pDeviceContext.Get() == nullptr);
				_M_pDevice->GetImmediateContext(_M_pDeviceContext.GetAddressOf());
				u->Release();

				_M_dx_type = _Dx_type;

				// Create an FFT interface and get buffer into
				UINT totalElementSize = 1;
				D3DX11_FFT_BUFFER_INFO fft_buffer_info;
				{
					D3DX11_FFT_DESC fft_desc;
					ZeroMemory(&fft_desc, sizeof(fft_desc));
					fft_desc.NumDimensions = _Dim;
					for(int i = 0; i < _Dim; i++)
					{
						totalElementSize *= _Transform_extent[i];
						fft_desc.ElementLengths[_Dim - 1 - i] = _Transform_extent[i];
					}
					fft_desc.DimensionMask = (1 << _Dim) - 1;
					fft_desc.Type = _Dx_type;
					ZeroMemory(&fft_buffer_info, sizeof(fft_buffer_info));
					HRESULT hr = D3DX11CreateFFT(_M_pDeviceContext.Get(), &fft_desc, 0, &fft_buffer_info, _M_pFFT.GetAddressOf());
					if(FAILED(hr)) throw fft_exception("Failed in fft constructor", hr);

					if(_Forward_scale != 0.0f)
						set_forward_scale(_Forward_scale);

					if(_Inverse_scale != 0.0f)
						set_inverse_scale(_Inverse_scale);
				}

				// Make sure we have at least two buffers that are big enough for input/output
				if(fft_buffer_info.NumTempBufferSizes < 2)
				{
					for(UINT i = fft_buffer_info.NumTempBufferSizes; i < 2; i++)
					{
						fft_buffer_info.NumTempBufferSizes = 0;
					}
					fft_buffer_info.NumTempBufferSizes = 2;
				}
				UINT elementFloatSize = (_Dx_type == D3DX11_FFT_DATA_TYPE_COMPLEX) ? 2 : 1;
				_M_Total_float_size = totalElementSize * elementFloatSize;
				for(UINT i = 0; i < 2; i++)
				{
					if(fft_buffer_info.TempBufferFloatSizes[i] < _M_Total_float_size)
						fft_buffer_info.TempBufferFloatSizes[i] = _M_Total_float_size;
				}

				ID3D11UnorderedAccessView *tempUAVs[D3DX11_FFT_MAX_TEMP_BUFFERS] = { 0 };
				ID3D11UnorderedAccessView *precomputeUAVs[D3DX11_FFT_MAX_PRECOMPUTE_BUFFERS] = { 0 };

				// Allocate temp and pre-computed buffers
				for(UINT i = 0; i < fft_buffer_info.NumTempBufferSizes; i++)
				{
					Microsoft::WRL::ComPtr<ID3D11Buffer> pBuffer;
					_ASSERTE(_M_pTempUAVs[i].Get() == nullptr);
					hr = create_raw_buffer_and_uav(fft_buffer_info.TempBufferFloatSizes[i], pBuffer.GetAddressOf(), _M_pTempUAVs[i].GetAddressOf());
					tempUAVs[i] = _M_pTempUAVs[i].Get();

					if(FAILED(hr)) goto cleanup;
				}

				for(UINT i = 0; i < fft_buffer_info.NumPrecomputeBufferSizes; i++)
				{
					Microsoft::WRL::ComPtr<ID3D11Buffer> pBuffer;
					_ASSERTE(_M_pPrecomputeUAVs[i].Get() == nullptr);
					hr = create_raw_buffer_and_uav(fft_buffer_info.PrecomputeBufferFloatSizes[i], pBuffer.GetAddressOf(), _M_pPrecomputeUAVs[i].GetAddressOf());
					precomputeUAVs[i] = _M_pPrecomputeUAVs[i].Get();

					if(FAILED(hr)) goto cleanup;
				}

				// Attach buffers and precompute

				hr = _M_pFFT->AttachBuffersAndPrecompute(
					fft_buffer_info.NumTempBufferSizes,
					&tempUAVs[0],
					fft_buffer_info.NumPrecomputeBufferSizes,
					&precomputeUAVs[0]);
				if(FAILED(hr)) goto cleanup;

			cleanup:
				if(FAILED(hr)) throw fft_exception("Failed in fft constructor", hr);
			}

			HRESULT base_transform(bool _Forward, ID3D11Buffer *pBufferIn, ID3D11Buffer *pBufferOut) const
			{
				HRESULT hr = S_OK;

				UINT inputFloatSize = _M_Total_float_size;
				UINT outputFloatSize = _M_Total_float_size;

				// When processing real numbers, if it is forward transform
				// then the output is twice the size of input as it produces
				// complex data and for inverse transform, the input is complex
				// data and hence twice the size of the output which are real numbers
				if(_M_dx_type != D3DX11_FFT_DATA_TYPE_COMPLEX)
				{
					if(_Forward) {
						outputFloatSize *= 2;
					}
					else {
						inputFloatSize *= 2;
					}
				}

				Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> pUAVIn, pUAVOut;

				hr = create_uav(inputFloatSize, pBufferIn, pUAVIn.GetAddressOf());
				if(FAILED(hr)) goto cleanup;

				hr = create_uav(outputFloatSize, pBufferOut, pUAVOut.GetAddressOf());
				if(FAILED(hr)) goto cleanup;

				// Do the transform
				if(_Forward)
				{
					hr = _M_pFFT->ForwardTransform(pUAVIn.Get(), pUAVOut.GetAddressOf());
				}
				else
				{
					hr = _M_pFFT->InverseTransform(pUAVIn.Get(), pUAVOut.GetAddressOf());
				}
				if(FAILED(hr)) goto cleanup;

			cleanup:
				return hr;
			}

		private:
			HRESULT create_raw_buffer(UINT floatSize, ID3D11Buffer **ppBuffer) const
			{
				D3D11_BUFFER_DESC desc;
				ZeroMemory(&desc, sizeof(desc));
				desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
				desc.ByteWidth = floatSize * sizeof(float);
				desc.CPUAccessFlags = 0;
				desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
				desc.Usage = D3D11_USAGE_DEFAULT;

				return _M_pDevice->CreateBuffer(&desc, NULL, ppBuffer);
			}

			HRESULT create_uav(UINT floatSize, ID3D11Buffer *ppBuffer, ID3D11UnorderedAccessView **ppUAV) const
			{
				D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
				ZeroMemory(&UAVDesc, sizeof(UAVDesc));
				UAVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
				UAVDesc.Buffer.FirstElement = 0;
				UAVDesc.Buffer.NumElements = floatSize;
				UAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

				return _M_pDevice->CreateUnorderedAccessView(ppBuffer, &UAVDesc, ppUAV);
			}

			HRESULT create_raw_buffer_and_uav(UINT floatSize, ID3D11Buffer **ppBuffer, ID3D11UnorderedAccessView **ppUAV) const
			{
				if(!ppBuffer || !ppUAV) return E_POINTER;
				if(*ppBuffer || *ppUAV) return E_INVALIDARG;

				HRESULT hr = S_OK;

				hr = create_raw_buffer(floatSize, ppBuffer);
				if(FAILED(hr)) goto cleanup;

				hr = create_uav(floatSize, *ppBuffer, ppUAV);
				if(FAILED(hr)) goto cleanup;

			cleanup:
				if(FAILED(hr))
				{
					if(*ppBuffer)
					{
						(*ppBuffer)->Release();
						*ppBuffer = NULL;
					}
					if(*ppUAV)
					{
						(*ppUAV)->Release();
						*ppUAV = NULL;
					}
				}
				return hr;
			}

			UINT _M_Total_float_size;
			D3DX11_FFT_DATA_TYPE _M_dx_type;
			Microsoft::WRL::ComPtr<ID3D11Device> _M_pDevice;
			Microsoft::WRL::ComPtr<ID3D11DeviceContext> _M_pDeviceContext;
			Microsoft::WRL::ComPtr<ID3DX11FFT> _M_pFFT;
			Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> _M_pTempUAVs[D3DX11_FFT_MAX_TEMP_BUFFERS];
			Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> _M_pPrecomputeUAVs[D3DX11_FFT_MAX_PRECOMPUTE_BUFFERS];
		};
	} // namespace _details

	//--------------------------------------------------------------------------------------
	// class fft.
	//
	// This is the class which provides the FFT transformation functionality. At this point
	// it exposes 1d, 2d and 3d transformations over floats or std::complex<float>, although
	// Direct3D also provides the ability to transform higher dimensions, or be more 
	// selective about which dimensions to transform. Such an extension is left for the 
	// future or to the interested reader...
	//
	// After creating an instance, you can use forward_transform and backward_tranform to
	// transform your data. The constructor initializes some internal data structures, so 
	// it's beneficial to reuse the fft object as long as possible.
	//
	// Note that the dimensions (extent) of the fft and the extent of all input and output 
	// arrays must be identical. If an array is used which has a different extent, an 
	// fft_exception is thrown.
	//
	// The library supports arbitrary extents but best performance can be achieved for 
	// powers of 2, followed by numbers whose prime factors are in the set {2,3,5}.
	//
	// Limitations:
	//
	//   -- You should only allocate one fft class per accelerator_view. If you need 
	//      additional fft classes, create additional accelerator_view's first. This is a 
	//      limitation of the Direct3D FFT API's.
	//
	//   -- Class fft is not thread safe. Or more accurately, the FFT API is not thread 
	//      safe. So again, create additional fft objects such that each thread has its own.
	//
	//--------------------------------------------------------------------------------------
	template <typename _Element_type, int _Dim>
	class fft : public _details::fft_base
	{
	private:
		static_assert(_Dim >= 1 && _Dim <= 3, "class fft is only available for one, two or three dimensions");
		static_assert(_details::dx_fft_type_helper<_Element_type>::is_type_supported, "class fft only supports element types float and std::complex<float>");

	public:
		//--------------------------------------------------------------------------------------
		// Constructor. Throws fft_exception on failure.
		//--------------------------------------------------------------------------------------
		fft(
			concurrency::extent<_Dim> _Transform_extent,
			const concurrency::accelerator_view& _Av = concurrency::accelerator().default_view,
			float _Forward_scale = 0.0f,
			float _Inverse_scale = 0.0f)
			:extent(_Transform_extent),
			fft_base(
			_details::dx_fft_type_helper<_Element_type>::dx_type,
			_Dim,
			&_Transform_extent[0],
			_Av,
			_Forward_scale,
			_Inverse_scale)
		{
		}

		//--------------------------------------------------------------------------------------
		// Forward transform. 
		//  -- Throws fft_exception on failure. 
		//  -- Arrays extents must be identical to those of the fft object.
		//  -- It is permissible for the input and output arrays to be references to the same 
		//     array.
		//--------------------------------------------------------------------------------------
		void forward_transform(const concurrency::array<_Element_type, _Dim>& input, concurrency::array<std::complex<typename _details::dx_fft_type_helper<_Element_type>::precision_type>, _Dim>& output) const
		{
			transform(true, input, output);
		}

		//--------------------------------------------------------------------------------------
		// Inverse transform. 
		//  -- Throws fft_exception on failure. 
		//  -- Arrays extents must be identical to those of the fft object.
		//  -- It is permissible for the input and output arrays to be references to the same 
		//     array.
		//--------------------------------------------------------------------------------------
		void inverse_transform(const concurrency::array<std::complex<typename _details::dx_fft_type_helper<_Element_type>::precision_type>, _Dim>& input, concurrency::array<_Element_type, _Dim>& output) const
		{
			transform(false, input, output);
		}

		//--------------------------------------------------------------------------------------
		// The extent of the fft transform.
		//--------------------------------------------------------------------------------------
		const concurrency::extent<_Dim> extent;

	private:

		template <typename _Input_element_type, typename _Output_element_type>
		void transform(bool _Forward, const concurrency::array<_Input_element_type, _Dim>& _Input, concurrency::array<_Output_element_type, _Dim>& _Output) const
		{
			if (_Input.get_extent() != extent)
				throw fft_exception("The input extent in transform is invalid", E_INVALIDARG);

			if (_Output.get_extent() != extent)
				throw fft_exception("The output extent in transform is invalid", E_INVALIDARG);

			HRESULT hr = S_OK;

			Microsoft::WRL::ComPtr<ID3D11Buffer> pBufferIn;
			direct3d::get_buffer(_Input)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(pBufferIn.GetAddressOf()));

			Microsoft::WRL::ComPtr<ID3D11Buffer> pBufferOut;
			direct3d::get_buffer(_Output)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(pBufferOut.GetAddressOf()));

			hr = base_transform(_Forward, pBufferIn.Get(), pBufferOut.Get());

			if (FAILED(hr)) throw fft_exception("transform failed", hr);
		}
	};

	// 2D Convolution based on FFT
	class fft_context
	{
	public:
		fft_context(const accelerator_view& acc_view_, concurrency::extent<2> dft_size_)
			: acc_view(acc_view_), dft_size(dft_size_), fft_obj(dft_size_, acc_view_)
			, image_block(dft_size_, acc_view_), kernel_block(dft_size_, acc_view_), result_data(dft_size_, acc_view_)
			, image_spect(dft_size_, acc_view_), kernel_spect(dft_size_, acc_view_), result_spect(dft_size_, acc_view_)
		{
		}

		static concurrency::extent<2> get_dft_size(concurrency::extent<2> image_size, concurrency::extent<2> kernel_size)
		{
			return concurrency::extent<2>(cv::getOptimalDFTSize(image_size[0] + kernel_size[0] - 1), cv::getOptimalDFTSize(image_size[1] + kernel_size[1] - 1));
		}

		concurrency::extent<2> dft_size;
		concurrency::array<std::complex<float>, 2> image_spect, kernel_spect, result_spect;
		concurrency::array<float, 2> image_block, kernel_block, result_data;
		accelerator_view acc_view;
		fft<float, 2> fft_obj;
	};

	inline void mul_scale_spectrum_32f_c1(const accelerator_view& acc_view, array_view<float_2, 2> src_array1, array_view<float_2, 2> src_array2, array_view<float_2, 2> dest_array, bool is_correlation = false)
	{
		static const int tile_size = 32;
		if(is_correlation)
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				float_2 a = guarded_read(src_array1, gidx);
				float_2 b = guarded_read(src_array2, gidx);
				guarded_write(dest_array, gidx, a * (b.x - b.y));
			});
		}
		else
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				float_2 a = guarded_read(src_array1, gidx);
				float_2 b = guarded_read(src_array2, gidx);
				guarded_write(dest_array, gidx, float_2(direct3d::mad(a.x, b.x, -a.y * b.y), direct3d::mad(a.x, b.y, a.y * b.x)));
			});
		}
	}

	inline void mul_scale_spectrum_32f_c1(const accelerator_view& acc_view, array_view<float_2, 1> src_array1, array_view<float_2, 1> src_array2, array_view<float_2, 1> dest_array, bool is_correlation = false)
	{
		static const int tile_size = 256;
		if(is_correlation)
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
			{
				concurrency::index<1> gidx = idx.global;
				float_2 a = guarded_read(src_array1, gidx);
				float_2 b = guarded_read(src_array2, gidx);
				guarded_write(dest_array, gidx, a * (b.x - b.y));
			});
		}
		else
		{
			parallel_for_each(acc_view, dest_array.get_extent().tile<tile_size>().pad(), [=](tiled_index<tile_size> idx) restrict(amp)
			{
				concurrency::index<1> gidx = idx.global;
				float_2 a = guarded_read(src_array1, gidx);
				float_2 b = guarded_read(src_array2, gidx);
				guarded_write(dest_array, gidx, float_2(direct3d::mad(a.x, b.x, -a.y * b.y), direct3d::mad(a.x, b.y, a.y * b.x)));
			});
		}
	}

	inline void convolve2d_fft_32f_c1(fft_context& ctx, array_view<const float, 2> src_array, array_view<float, 2> dest_array, array_view<const float, 2> kernel, bool reuse_src = false, bool reuse_kernel = false)
	{
		// std::cout << "dft size: " << ctx.dft_size[1] << 'x' << ctx.dft_size[0] << std::endl;
		static const int tile_size = 32;
		concurrency::array_view<float, 2> image_block_view(ctx.image_block);
		concurrency::array_view<float, 2> kernel_block_view(ctx.kernel_block);
		concurrency::array_view<float, 2> result_data_view(ctx.result_data);
		// 1.pad kernel and image
		if(!reuse_src)
		{
			concurrency::extent<2> padded_image_ext(src_array.get_extent()[0] + kernel.get_extent()[0] - 1, src_array.get_extent()[1] + kernel.get_extent()[1] - 1);
			int expand_rows = (padded_image_ext[0] - src_array.get_extent()[0]) / 2;
			int expand_cols = (padded_image_ext[1] - src_array.get_extent()[1]) / 2;
			parallel_for_each(ctx.acc_view, image_block_view.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
			{
				concurrency::index<2> gidx = idx.global;
				float value = padded_image_ext.contains(gidx) ? guarded_read_reflect101(src_array, concurrency::index<2>(gidx[0] - expand_rows, gidx[1] - expand_cols)) : 0.0f;
				guarded_write(image_block_view, gidx, value);
			});
		}
		if(!reuse_kernel)
		{
			copy_make_border_32f_c1(ctx.acc_view, kernel, kernel_block_view, amp::border_type::constant, 0.0f, 0, 0);
		}
		// 2.dft kernel and image
		if(!reuse_src)
		{
			ctx.fft_obj.forward_transform(ctx.image_block, ctx.image_spect);
		}
		if(!reuse_kernel)
		{
			ctx.fft_obj.forward_transform(ctx.kernel_block, ctx.kernel_spect);
		}
		// 3.multiply
		mul_scale_spectrum_32f_c1(ctx.acc_view, ctx.image_spect.reinterpret_as<float_2>(), ctx.kernel_spect.reinterpret_as<float_2>()
			, ctx.result_spect.reinterpret_as<float_2>(), false);
		// 4.inverse transfer result
		ctx.fft_obj.inverse_transform(ctx.result_spect, ctx.result_data);
		// 5.copy to dest array
		array_view<float, 2> result_block = result_data_view.section(kernel.get_extent()[0] - 1, kernel.get_extent()[1] - 1, dest_array.get_extent()[0], dest_array.get_extent()[1]);
		dest_array.discard_data();
		parallel_for_each(ctx.acc_view, dest_array.get_extent().tile<tile_size, tile_size>().pad(), [=](tiled_index<tile_size, tile_size> idx) restrict(amp)
		{
			guarded_write(dest_array, idx.global, guarded_read(result_block, idx.global));
		});
	}

	inline void convolve2d_fft_32f_c1(fft_context& ctx, array_view<const float, 2> src_array, array_view<float, 2> dest_array, const cv::Mat& kernel, bool reuse_src = false, bool reuse_kernel = false)
	{
		cv::Mat convertedKernel;
		assert(kernel.channels() == 1);
		if(!kernel.isContinuous() || kernel.depth() != CV_32F)
		{
			kernel.convertTo(convertedKernel, CV_32F);
		}
		else
		{
			kernel.copyTo(convertedKernel);
		}
		concurrency::array<float, 2> gpuKernel(convertedKernel.rows, convertedKernel.cols, ctx.acc_view);
		concurrency::copy(convertedKernel.ptr<float>(0), gpuKernel);
		convolve2d_fft_32f_c1(ctx, src_array, dest_array, gpuKernel, reuse_src, reuse_kernel);
	}

} // namespace amp
