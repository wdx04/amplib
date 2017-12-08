# amplib
70+ Computer Vision algorithms written in C++ AMP, many are transformed from OpenCV's OpenCL code.

Requirements:
Hardware: AMD GCN GPU or NVIDIA Maxwell/Pascal/Volta GPU recommended
Software: Windows/Visual Studio 2013 or later(may or may not work on Linux/AMD HCC)
          OpenCV

Performance:
Acceptable for non-iterative algorithms like thresholding, gaussian filter, guided filter, etc.
Bad for iterative algorithms, because C++ AMP does not support device-side enqueue.

Example:

```C++
#include <amp_core.h>
#include <amp_delete_border_components.h>

// read an image from disk
cv::Mat src = cv::imread("test.png");
// create accelerator view
concurrency::accelerator_view acc_view = concurrency::accelerator().create_view();
// create vision context on accelerator view(second param is the size of shared buffer for saving/loading cv::Mats, in MB)
amp::vision_context ctx(acc_view, src.rows * src.cols / 1.0e6);
// create a few arrays on GPU(you can also create concurrency::arrays manually)
ctx.create_float2d_buf(src.rows, src.cols, 3);
// load cv::Mat into a GPU array(support 8UC1 & 8UC3, 16UC1 & 16UC3 formats)
ctx.load_cv_mat(src, ctx.float2d[0]);
// run algorithms on GPU
amp::delete_border_components_32f_c1(acc_view, ctx.float2d[0], ctx.float2d[1], ctx.float2d[2]);
//  save results to a cv::Mat
cv::Mat dest;
ctx.save_cv_mat(ctx.float2d[1], dest);
```
