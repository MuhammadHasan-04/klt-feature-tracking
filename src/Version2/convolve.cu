/*********************************************************************
 * convolve.cu - CUDA implementation
 *********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "convolve.h"

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }


// CUDA kernel for horizontal convolution
__global__ void convolveHorizontalKernel(
    float* input, float* output,
    float* kernel, int kernel_width,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  

    if (idx < width && idy < height) {
        int radius = kernel_width / 2;
        float sum = 0.0f;

        if (idx < radius) {
            output[idy * width + idx] = 0.0f;
        }

        if (idx >= width - radius) {
            output[idy * width + idx] = 0.0f;
        }

        if (idx >= radius && idx < width - radius) {
            for (int k = 0; k < kernel_width; k++) {
                int x = idx - radius + k;
                sum += input[idy * width + x] * kernel[kernel_width - 1 - k];
            }
            output[idy * width + idx] = sum;
        }
    }
}

// CUDA kernel for vertical convolution
__global__ void convolveVerticalKernel(
    float* input, float* output,
    float* kernel, int kernel_width,
    int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y; 

    if (idx < width && idy < height) {
        int radius = kernel_width / 2;
        float sum = 0.0f;

        if (idy < radius) {
            output[idy * width + idx] = 0.0f;
        }

        if (idy >= height - radius) {
            output[idy * width + idx] = 0.0f;
        }

        if (idy >= radius && idy < height - radius) {
            for (int k = 0; k < kernel_width; k++) {
                int y = idy - radius + k;
                sum += input[y * width + idx] * kernel[kernel_width - 1 - k];
            }
            output[idy * width + idx] = sum;
        }
    }
}

extern "C" void gpuConvolve(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    int width  = imgin->ncols;
    int height = imgin->nrows;
    size_t imgSize = (size_t)width * height * sizeof(float);

    float *d_input = nullptr, *d_temp = nullptr, *d_output = nullptr;
    float *d_hKernel = nullptr, *d_vKernel = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input,  imgSize));
    CHECK_CUDA(cudaMalloc(&d_temp,   imgSize));
    CHECK_CUDA(cudaMalloc(&d_output, imgSize));
    CHECK_CUDA(cudaMalloc(&d_hKernel, horiz_kernel.width * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vKernel, vert_kernel.width * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, imgin->data, imgSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_hKernel, horiz_kernel.data,
                          horiz_kernel.width * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vKernel, vert_kernel.data,
                          vert_kernel.width * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    convolveHorizontalKernel<<<grid, block>>>(d_input, d_temp, d_hKernel,
                                              horiz_kernel.width, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    convolveVerticalKernel<<<grid, block>>>(d_temp, d_output, d_vKernel,
                                            vert_kernel.width, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(imgout->data, d_output, imgSize, cudaMemcpyDeviceToHost));

    imgout->ncols = width;
    imgout->nrows = height;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_hKernel));
    CHECK_CUDA(cudaFree(d_vKernel));
}