#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "klt.h"
#include "base.h"
#include "error.h"

#define CUDA_CHECK(call)                                  \
    {                                                     \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess)                           \
        {                                                 \
            fprintf(stderr, "CUDA Error: %s (line %d)\n", \
                    cudaGetErrorString(err), __LINE__);   \
            cudaDeviceReset();                            \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    }

    

__global__ void featureStrengthComputation(
    const float *gradx, const float *grady, float *strength,
    int width, int height, int win_half_w, int win_half_h,
    int borderx, int bordery)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < borderx)
        return;

    if (x >= width - borderx)
        return;

    if (y < bordery)
        return;

    if (y >= height - bordery)
        return;

    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;

    for (int j = -win_half_h; j <= win_half_h; j++)
    {
        for (int i = -win_half_w; i <= win_half_w; i++)
        {
            int idx = (y + j) * width + (x + i);
            float gxVal = gradx[idx];
            float gyVal = grady[idx];

            gxx += gxVal * gxVal;
            gxy += gxVal * gyVal;
            gyy += gyVal * gyVal;
        }
    }

    float trace = gxx + gyy;
    float diff = gxx - gyy;
    float temp = sqrtf(diff * diff + 4.0f * gxy * gxy);
    float lambda_min = 0.5f * (trace - temp);

    strength[y * width + x] = lambda_min;
}

// Wrapper function to handle GPU-based feature strength computation
extern "C" void gpuComputeFeatureStrength(
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady,
    _KLT_FloatImage strength,
    int win_width,
    int win_height,
    int borderx,
    int bordery)
{
    int width = gradx->ncols;
    int height = gradx->nrows;
    size_t bufferSize = static_cast<size_t>(width * height * sizeof(float));

    float *dev_gradx = nullptr;
    float *dev_grady = nullptr;
    float *dev_strength = nullptr;

    printf("Launching GPU kernel for corner strength estimation...\n");

    CUDA_CHECK(cudaMalloc(&dev_gradx, bufferSize));
    CUDA_CHECK(cudaMalloc(&dev_grady, bufferSize));
    CUDA_CHECK(cudaMalloc(&dev_strength, bufferSize));

    CUDA_CHECK(cudaMemcpy(dev_gradx, gradx->data, bufferSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_grady, grady->data, bufferSize, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t evtStart, evtEnd;
    CUDA_CHECK(cudaEventCreate(&evtStart));
    CUDA_CHECK(cudaEventCreate(&evtEnd));
    CUDA_CHECK(cudaEventRecord(evtStart));

    featureStrengthComputation<<<numBlocks, threadsPerBlock>>>(
        dev_gradx, dev_grady, dev_strength,
        width, height,
        win_width / 2, win_height / 2,
        borderx, bordery);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(evtEnd));
    CUDA_CHECK(cudaEventSynchronize(evtEnd));

    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, evtStart, evtEnd));
    printf("GPU execution finished in %.3f ms\n", gpuTime);

    CUDA_CHECK(cudaMemcpy(strength->data, dev_strength, bufferSize, cudaMemcpyDeviceToHost));

    cudaFree(dev_gradx);
    cudaFree(dev_grady);
    cudaFree(dev_strength);

    CUDA_CHECK(cudaEventDestroy(evtStart));
    CUDA_CHECK(cudaEventDestroy(evtEnd));
    CUDA_CHECK(cudaDeviceSynchronize());
}
