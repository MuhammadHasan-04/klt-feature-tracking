/*********************************************************************
 * trackFeats.cu
 * 
 * CUDA-accelerated gradient computation functions only
 * DO NOT include KLTTrackFeatures here
 *********************************************************************/

#include <assert.h>
#include <math.h>       /* floorf, fabs */
#include <stdlib.h>     /* malloc, free */
#include <stdio.h>      /* fprintf, fflush */
#include <string.h>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"   /* _KLT_FloatImage */
#include "pyramid.h"    /* _KLT_Pyramid */

extern int KLT_verbose;
typedef float *_FloatWindow;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* -------------------------
   Persistent cache struct
   ------------------------- */
typedef struct {
    float *d_ptr;     // device pointer
    int width;        // image width
    int height;       // image height
    size_t capacity;  // allocated elements (width*height)
    bool valid;
} ImgCache;

/* Static persistent caches for the four gradient images */
static ImgCache cache_gx1 = {0,0,0,0,false};
static ImgCache cache_gy1 = {0,0,0,0,false};
static ImgCache cache_gx2 = {0,0,0,0,false};
static ImgCache cache_gy2 = {0,0,0,0,false};

/* Static stream to avoid creating stream per call */
static cudaStream_t cached_stream = NULL;
static bool stream_initialized = false;

/* Helper to ensure a device buffer is allocated with at least width*height capacity */
static void device_Image_ensured(ImgCache *c, int width, int height) {
    size_t total_elements = (size_t)width * (size_t)height;
    if (c->valid && c->width == width && c->height == height && c->capacity >= total_elements) {
        // already allocated and fits
        return;
    }
    // if existing allocation is insufficient, free and reallocate
    if (c->valid && c->d_ptr != NULL) {
        CUDA_CHECK(cudaFree(c->d_ptr));
        c->d_ptr = NULL;
        c->valid = false;
    }
    // allocate new
    float *device_memory = NULL;
    CUDA_CHECK(cudaMalloc((void**)&device_memory, total_elements * sizeof(float)));
    c->d_ptr = device_memory;
    c->width = width;
    c->height = height;
    c->capacity = total_elements;
    c->valid = true;
}

// simple bilinear sampler for float images (width = img_w, height = img_h)
// returns 0 for out-of-bounds coordinates
static __device__ inline float sampleBilinear(const float *img, int img_w, int img_h, float fx, float fy) {
    if (fx < 0.f || fy < 0.f || fx > (float)(img_w-1) || fy > (float)(img_h-1)) return 0.0f;
    int base_x = (int)floorf(fx);
    int base_y = (int)floorf(fy);
    int next_x = min(base_x + 1, img_w - 1);
    int next_y = min(base_y + 1, img_h - 1);
    float offset_x = fx - (float)base_x;
    float offset_y = fy - (float)base_y;
    float value_00 = img[base_y * img_w + base_x];
    float value_10 = img[base_y * img_w + next_x];
    float value_01 = img[next_y * img_w + base_x];
    float value_11 = img[next_y * img_w + next_x];
    float interpolated_bottom = value_00 * (1.0f - offset_x) + value_10 * offset_x;
    float interpolated_top = value_01 * (1.0f - offset_x) + value_11 * offset_x;
    return interpolated_bottom * (1.0f - offset_y) + interpolated_top * offset_y;
}


extern "C" __global__
void _computeGradientSumKernelBatched(
    const float *d_gx1, const float *d_gy1,
    const float *d_gx2, const float *d_gy2,
    const float *d_x1, const float *d_y1,
    const float *d_x2, const float *d_y2,
    int img_w, int img_h,
    int win_w, int win_h,
    float *d_out_gx_all, float *d_out_gy_all,
    int batchN)
{
    int feature_index = blockIdx.x;                // feature/window index
    if (feature_index >= batchN) return;

    // per-window offsets in output arrays
    int window_pixels = win_w * win_h;
    float *output_gx = d_out_gx_all + (size_t)feature_index * window_pixels;
    float *output_gy = d_out_gy_all + (size_t)feature_index * window_pixels;

    // get centers
    float center_x1 = d_x1[feature_index];
    float center_y1 = d_y1[feature_index];
    float center_x2 = d_x2[feature_index];
    float center_y2 = d_y2[feature_index];

    // thread coordinates inside block
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_size_x = blockDim.x; int block_size_y = blockDim.y;

    // iterate over window pixels by mapping (u,v) = tx + i*bx, ty + j*by
    for (int row = thread_y; row < win_h; row += block_size_y) {
        for (int col = thread_x; col < win_w; col += block_size_x) {
            // compute sample positions relative to window center
            // original code uses integer offsets i,j from -hw..hw; match that:
            int half_width = win_w / 2;
            int half_height = win_h / 2;
            float sample_x1 = center_x1 + (float)(col - half_width);
            float sample_y1 = center_y1 + (float)(row - half_height);
            float sample_x2 = center_x2 + (float)(col - half_width);
            float sample_y2 = center_y2 + (float)(row - half_height);

            float grad_x1 = sampleBilinear(d_gx1, img_w, img_h, sample_x1, sample_y1);
            float grad_y1 = sampleBilinear(d_gy1, img_w, img_h, sample_x1, sample_y1);
            float grad_x2 = sampleBilinear(d_gx2, img_w, img_h, sample_x2, sample_y2);
            float grad_y2 = sampleBilinear(d_gy2, img_w, img_h, sample_x2, sample_y2);

            int pixel_index = row * win_w + col;
            output_gx[pixel_index] = grad_x1 + grad_x2; // original "gradient sum" semantics: sum across images
            output_gy[pixel_index] = grad_y1 + grad_y2;
        }
    }
}

__global__ void _computeGradientSumKernelGlobal(
    const float* __restrict__ gradx1,
    const float* __restrict__ grady1,
    const float* __restrict__ gradx2,
    const float* __restrict__ grady2,
    float x1, float y1, float x2, float y2,
    int img_width, int img_height,
    int win_w, int win_h,
    float* gradx_out, float* grady_out)
{
    // thread-local within window
    int local_col = blockIdx.x * blockDim.x + threadIdx.x; // column in window [0, win_w)
    int local_row = blockIdx.y * blockDim.y + threadIdx.y; // row in window [0, win_h)

    if (local_col >= win_w || local_row >= win_h) return;

    int col_offset = local_col - (win_w / 2);
    int row_offset = local_row - (win_h / 2);

    // sample coords in image space
    float sample_x1 = x1 + (float)col_offset;
    float sample_y1 = y1 + (float)row_offset;
    float sample_x2 = x2 + (float)col_offset;
    float sample_y2 = y2 + (float)row_offset;

    // helper lambda-like logic inline for bilinear interpolation with boundary checks
    auto sample_bilinear = [&](const float* img, float sample_x, float sample_y) -> float {
        // if outside or at border where neighbors unavailable, return 0
        // we want valid ix in [0, img_width-2] and iy in [0, img_height-2] for 4-neighbor sample
        if (sample_x < 0.0f || sample_y < 0.0f || sample_x >= (float)(img_width - 1) || sample_y >= (float)(img_height - 1)) {
            // clamp strategy instead of returning 0 might be used, but original code returned 0 for out-of-bounds
            return 0.0f;
        }
        int base_x = (int)floorf(sample_x);
        int base_y = (int)floorf(sample_y);
        float x_frac = sample_x - (float)base_x;
        float y_frac = sample_y - (float)base_y;

        // compute indices for four neighbors, read from global memory (coalesced if accesses aligned)
        int base_index = base_y * img_width + base_x;
        float value_00 = img[base_index];
        float value_01 = img[base_index + 1];
        float value_10 = img[base_index + img_width];
        float value_11 = img[base_index + img_width + 1];

        // bilinear
        return (1.0f - x_frac) * (1.0f - y_frac) * value_00
             + x_frac * (1.0f - y_frac) * value_01
             + (1.0f - x_frac) * y_frac * value_10
             + x_frac * y_frac * value_11;
    };

    float grad_x1 = sample_bilinear(gradx1, sample_x1, sample_y1);
    float grad_y1 = sample_bilinear(grady1, sample_x1, sample_y1);
    float grad_x2 = sample_bilinear(gradx2, sample_x2, sample_y2);
    float grad_y2 = sample_bilinear(grady2, sample_x2, sample_y2);

    int output_index = local_row * win_w + local_col;
    gradx_out[output_index] = grad_x1 + grad_x2;
    grady_out[output_index] = grad_y1 + grad_y2;
}

/* Extern "C" wrapper - signature preserved */
#ifdef __cplusplus
extern "C" {
#endif

void _compute2by2GradientMatrixRaw(float *gradx, float *grady, int w, int h,
                                   float *gxx, float *gxy, float *gyy) {
    *gxx = *gxy = *gyy = 0.0f;
    for (int i = 0; i < w * h; i++) {
        *gxx += gradx[i] * gradx[i];
        *gxy += gradx[i] * grady[i];
        *gyy += grady[i] * grady[i];
    }
}

void _compute2by1ErrorVectorRaw(float *gradx, float *grady, int w, int h,
                                float step_factor, float *ex, float *ey) {
    *ex = *ey = 0.0f;
    for (int i = 0; i < w * h; i++) {
        *ex += gradx[i] * step_factor;
        *ey += grady[i] * step_factor;
    }
}

void _computeGradientSum_CUDA_batched(
    _KLT_FloatImage gradx1, _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2, _KLT_FloatImage grady2,
    const float *features_x1, const float *features_y1,
    const float *features_x2, const float *features_y2,
    int count, int win_w, int win_h,
    _FloatWindow out_gradx_all, _FloatWindow out_grady_all)
{
    if (count <= 0) return;

    const int img_w = gradx1->ncols;
    const int img_h = gradx1->nrows;
    const size_t img_elems = (size_t)img_w * img_h;
    const size_t img_bytes = img_elems * sizeof(float);
    const int win_elems = win_w * win_h;
    const size_t batch_out_bytes = (size_t)win_elems * count * sizeof(float);

    // static cached allocations to avoid repeated cudaMalloc/free
    static float *d_gx1 = NULL, *d_gy1 = NULL, *d_gx2 = NULL, *d_gy2 = NULL;
    static size_t d_img_capacity = 0;
    static float *d_x1 = NULL, *d_y1 = NULL, *d_x2 = NULL, *d_y2 = NULL;
    static float *d_out_gx_all = NULL, *d_out_gy_all = NULL;
    static size_t d_out_capacity = 0;

    static float *h_x1_pinned = NULL, *h_y1_pinned = NULL, *h_x2_pinned = NULL, *h_y2_pinned = NULL;
    static float *h_out_gx_pinned = NULL, *h_out_gy_pinned = NULL;
    static size_t h_coords_capacity = 0, h_out_capacity = 0;

    static cudaStream_t stream = 0;
    static bool stream_inited = false;
    if (!stream_inited) { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); stream_inited = true; }

    // allocate device image buffers once (each holds one image)
    size_t required_img_bytes = img_bytes; // single image
    if (d_img_capacity < required_img_bytes) {
        // free previous
        if (d_gx1) { cudaFree(d_gx1); cudaFree(d_gy1); cudaFree(d_gx2); cudaFree(d_gy2); }
        cudaMalloc((void**)&d_gx1, required_img_bytes);
        cudaMalloc((void**)&d_gy1, required_img_bytes);
        cudaMalloc((void**)&d_gx2, required_img_bytes);
        cudaMalloc((void**)&d_gy2, required_img_bytes);
        d_img_capacity = required_img_bytes;
    }


    cudaMemcpyAsync(d_gx1, gradx1->data, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gy1, grady1->data, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gx2, gradx2->data, img_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gy2, grady2->data, img_bytes, cudaMemcpyHostToDevice, stream);

    size_t coords_bytes = (size_t)count * sizeof(float);
    if (h_coords_capacity < coords_bytes) {
        if (h_x1_pinned) { cudaFreeHost(h_x1_pinned); cudaFreeHost(h_y1_pinned); cudaFreeHost(h_x2_pinned); cudaFreeHost(h_y2_pinned); }
        cudaMallocHost((void**)&h_x1_pinned, coords_bytes);
        cudaMallocHost((void**)&h_y1_pinned, coords_bytes);
        cudaMallocHost((void**)&h_x2_pinned, coords_bytes);
        cudaMallocHost((void**)&h_y2_pinned, coords_bytes);
        h_coords_capacity = coords_bytes;
    }

    memcpy(h_x1_pinned, features_x1, coords_bytes);
    memcpy(h_y1_pinned, features_y1, coords_bytes);
    memcpy(h_x2_pinned, features_x2, coords_bytes);
    memcpy(h_y2_pinned, features_y2, coords_bytes);

    // allocate device coordinate arrays sized for count
    size_t coord_dev_bytes = coords_bytes;
    if (!d_x1 || d_out_capacity < (size_t)count) { // lazy allocate or reallocate
        if (d_x1) { cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_x2); cudaFree(d_y2); }
        cudaMalloc((void**)&d_x1, coord_dev_bytes);
        cudaMalloc((void**)&d_y1, coord_dev_bytes);
        cudaMalloc((void**)&d_x2, coord_dev_bytes);
        cudaMalloc((void**)&d_y2, coord_dev_bytes);
    }

    // async H->D coords
    cudaMemcpyAsync(d_x1, h_x1_pinned, coord_dev_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y1, h_y1_pinned, coord_dev_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x2, h_x2_pinned, coord_dev_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y2, h_y2_pinned, coord_dev_bytes, cudaMemcpyHostToDevice, stream);

    // allocate device outputs for all windows if needed
    size_t required_out_bytes = batch_out_bytes;
    if (d_out_capacity < required_out_bytes) {
        if (d_out_gx_all) { cudaFree(d_out_gx_all); cudaFree(d_out_gy_all); }
        cudaMalloc((void**)&d_out_gx_all, required_out_bytes);
        cudaMalloc((void**)&d_out_gy_all, required_out_bytes);
        d_out_capacity = required_out_bytes;
    }

    // allocate pinned host output buffers if needed
    if (h_out_capacity < required_out_bytes) {
        if (h_out_gx_pinned) { cudaFreeHost(h_out_gx_pinned); cudaFreeHost(h_out_gy_pinned); }
        cudaMallocHost((void**)&h_out_gx_pinned, required_out_bytes);
        cudaMallocHost((void**)&h_out_gy_pinned, required_out_bytes);
        h_out_capacity = required_out_bytes;
    }

    // Ensure device copies of images + coords are visible before kernel (they are on same stream)
    // Launch a single kernel with one block per window
    // Choose blockDim to cover typical window sizes; use 16x16 (works up to 16x16 windows without wasted loops)
    const int BLK_X = min(32, max(8, win_w)); // choose block dims safely (<= 32)
    const int BLK_Y = min(32, max(8, win_h));
    dim3 block(BLK_X, BLK_Y);
    dim3 grid(count, 1, 1);

    // single launch processes all windows in parallel
    _computeGradientSumKernelBatched<<<grid, block, 0, stream>>>(
        d_gx1, d_gy1, d_gx2, d_gy2,
        d_x1, d_y1, d_x2, d_y2,
        img_w, img_h, win_w, win_h,
        d_out_gx_all, d_out_gy_all,
        count);

    cudaMemcpyAsync(h_out_gx_pinned, d_out_gx_all, required_out_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out_gy_pinned, d_out_gy_all, required_out_bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Copy pinned host outputs into caller's output arrays (these are plain host buffers)
    memcpy(out_gradx_all, h_out_gx_pinned, required_out_bytes);
    memcpy(out_grady_all, h_out_gy_pinned, required_out_bytes);

    return;
}

void _computeGradientSum_CUDA(
    _KLT_FloatImage gradx1,
    _KLT_FloatImage grady1,
    _KLT_FloatImage gradx2,
    _KLT_FloatImage grady2,
    float x1, float y1,
    float x2, float y2,
    int win_w, int win_h,     // size of window
    _FloatWindow gradx,       // output pointers (host)
    _FloatWindow grady)
{
    /* Quick checks */
    if (!gradx1 || !grady1 || !gradx2 || !grady2) return;
    if (!gradx || !grady) return;
    if (win_w <= 0 || win_h <= 0) return;

    const int img_w = gradx1->ncols;
    const int img_h = gradx1->nrows;
    size_t img_elems = (size_t)img_w * (size_t)img_h;
    size_t img_bytes = img_elems * sizeof(float);
    size_t win_elems = (size_t)win_w * (size_t)win_h;
    size_t win_bytes = win_elems * sizeof(float);

    // Static persistent device-side gradient buffer (4 images packed)
    static float *d_grad_pack = NULL;
    static size_t d_grad_pack_capacity = 0; // in bytes
    // Pointers into d_grad_pack for the four gradient images
    float *d_gx1 = NULL, *d_gy1 = NULL, *d_gx2 = NULL, *d_gy2 = NULL;

    // Static cached stream for asynchronous ops
    static cudaStream_t cached_stream = 0;
    static bool stream_init = false;

    // Host pinned staging buffer for packed gradients
    static float *h_grad_pack_pinned = NULL;
    static size_t h_grad_pack_capacity = 0; // in bytes

    // Static device output buffers (for window result) and pinned host buffers
    static float *d_out_gx = NULL, *d_out_gy = NULL;
    static float *h_out_gx_pinned = NULL, *h_out_gy_pinned = NULL;
    static size_t d_out_capacity = 0;
    static size_t h_out_capacity = 0;

    if (!stream_init) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&cached_stream, cudaStreamNonBlocking));
        stream_init = true;
    }

    // Ensure packed device gradient buffer has enough capacity (4 * img_bytes)
    size_t required_d_grad_bytes = img_bytes * 4;
    if (d_grad_pack_capacity < required_d_grad_bytes) {
        if (d_grad_pack) CUDA_CHECK(cudaFree(d_grad_pack));
        CUDA_CHECK(cudaMalloc((void**)&d_grad_pack, required_d_grad_bytes));
        d_grad_pack_capacity = required_d_grad_bytes;
    }
    // Set per-image device pointers as slices of packed buffer
    d_gx1 = d_grad_pack + 0 * img_elems;
    d_gy1 = d_grad_pack + 1 * img_elems;
    d_gx2 = d_grad_pack + 2 * img_elems;
    d_gy2 = d_grad_pack + 3 * img_elems;

    // Ensure host pinned staging buffer exists and is large enough
    size_t required_h_grad_bytes = required_d_grad_bytes;
    if (h_grad_pack_capacity < required_h_grad_bytes) {
        if (h_grad_pack_pinned) CUDA_CHECK(cudaFreeHost(h_grad_pack_pinned));
        CUDA_CHECK(cudaMallocHost((void**)&h_grad_pack_pinned, required_h_grad_bytes));
        h_grad_pack_capacity = required_h_grad_bytes;
    }

    // Pack the four gradient images into the pinned host buffer (contiguous)
    // Note: source gradx1->data etc may be regular host memory
    float *hp = h_grad_pack_pinned;
    memcpy(hp + 0*img_elems, gradx1->data, img_bytes);
    memcpy(hp + 1*img_elems, grady1->data, img_bytes);
    memcpy(hp + 2*img_elems, gradx2->data, img_bytes);
    memcpy(hp + 3*img_elems, grady2->data, img_bytes);

    // Single H->D copy for all gradients into d_grad_pack
    CUDA_CHECK(cudaMemcpyAsync(d_grad_pack, h_grad_pack_pinned, required_h_grad_bytes, cudaMemcpyHostToDevice, cached_stream));

    // Ensure device output buffers (one per window) and pinned host output buffers are large enough
    if (d_out_capacity < win_bytes) {
        if (d_out_gx) CUDA_CHECK(cudaFree(d_out_gx));
        if (d_out_gy) CUDA_CHECK(cudaFree(d_out_gy));
        CUDA_CHECK(cudaMalloc((void**)&d_out_gx, win_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_out_gy, win_bytes));
        d_out_capacity = win_bytes;
    }
    if (h_out_capacity < win_bytes) {
        if (h_out_gx_pinned) CUDA_CHECK(cudaFreeHost(h_out_gx_pinned));
        if (h_out_gy_pinned) CUDA_CHECK(cudaFreeHost(h_out_gy_pinned));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_gx_pinned, win_bytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_gy_pinned, win_bytes));
        h_out_capacity = win_bytes;
    }

    // Launch kernel on cached_stream; kernel will use device slices d_gx1,d_gy1,d_gx2,d_gy2
    dim3 block(16, 16);
    dim3 grid( (win_w + block.x - 1) / block.x,
               (win_h + block.y - 1) / block.y );

    // Note: _computeGradientSumKernelGlobal expects pointers to image gradients
    _computeGradientSumKernelGlobal<<<grid, block, 0, cached_stream>>>(
        d_gx1, d_gy1, d_gx2, d_gy2,
        x1, y1, x2, y2,
        img_w, img_h,
        win_w, win_h,
        d_out_gx, d_out_gy);

    // Copy device window outputs -> pinned host buffers asynchronously
    CUDA_CHECK(cudaMemcpyAsync(h_out_gx_pinned, d_out_gx, win_bytes, cudaMemcpyDeviceToHost, cached_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_gy_pinned, d_out_gy, win_bytes, cudaMemcpyDeviceToHost, cached_stream));

    // Wait for operations on cached_stream to complete (only this stream)
    CUDA_CHECK(cudaStreamSynchronize(cached_stream));

    // Copy from pinned host buffers into user-provided output arrays (may be regular host memory)
    memcpy(gradx, h_out_gx_pinned, win_bytes);
    memcpy(grady, h_out_gy_pinned, win_bytes);

    // Do not free cached buffers; they persist for future calls to reduce overhead.
    return;
}

#ifdef __cplusplus
}
#endif