# GPU-Accelerated KLT Feature Tracking

This directory contains GPU-accelerated implementations of the KLT (Kanade-Lucas-Tomasi) feature tracking algorithm using NVIDIA CUDA.

## Overview

The GPU acceleration has been applied to the feature selection phase, specifically the computation of feature strength (minimum eigenvalue calculation). This is one of the most computationally intensive parts of the KLT algorithm.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA compute capability 8.6 or higher (e.g., RTX 30 series or newer)
- Sufficient GPU memory for image processing (typically 2GB+ recommended)

### Software Requirements
- NVIDIA CUDA Toolkit (11.0 or later recommended)
- GCC or compatible C compiler
- OpenCV libraries (opencv_core, opencv_imgproc, opencv_highgui, opencv_videoio)
- Make build system

## Building the Project

### Standard Build (CPU + GPU)
```bash
cd src/Version2
make clean
make lib          # Build the library with GPU support
make example1     # Build example programs
```

### Building Individual Examples
```bash
make example1    # Build example1
make example2    # Build example2
# ... and so on
```

## Running Examples

After building, you can run the examples:
```bash
./example1 ../../data/img0.pgm ../../data/img1.pgm
./example2 ../../data/img0.pgm ../../data/img1.pgm
./example3 ../../data/img0.pgm ../../data/img1.pgm
# etc.
```

## GPU Profiling

To profile GPU performance and measure execution time:

```bash
make profile_example1
make profile_example2
# etc.
```

This will:
1. Build the example with profiling enabled
2. Run the example with `nvprof` (NVIDIA profiler)
3. Generate a profile report file (e.g., `profile_example1.txt`)

### Alternative: Manual Profiling
You can also manually profile using NVIDIA tools:

```bash
# Using nvprof (legacy)
nvprof ./example1 ../../data/img0.pgm ../../data/img1.pgm

# Using newer nsys (Nsight Systems)
nsys profile ./example1 ../../data/img0.pgm ../../data/img1.pgm
```

## GPU Implementation Details

### Files Added
- **selectGoodFeatures.cu**: CUDA implementation of feature strength computation
- **selectGoodFeatures.h**: Header file declaring GPU functions
- **Makefile**: Updated to support CUDA compilation

### Key Functions

#### `gpuComputeFeatureStrength()`
Computes the feature strength map on the GPU.

**Parameters:**
- `gradx`: Gradient image in X direction
- `grady`: Gradient image in Y direction  
- `strength`: Output strength/trackability map
- `win_width`: Processing window width
- `win_height`: Processing window height
- `borderx`: Border to ignore in X direction
- `bordery`: Border to ignore in Y direction

**Performance:**
The GPU kernel uses a 16x16 thread block configuration and processes all pixels in parallel. Execution time is reported to the console.

### CUDA Kernel: `featureStrengthComputation`

This kernel computes the minimum eigenvalue of the structure tensor for each pixel:

1. Computes the local gradient covariance matrix (gxx, gxy, gyy)
2. Calculates the minimum eigenvalue using: `λ_min = 0.5 * (trace - sqrt(diff² + 4*gxy²))`
3. Stores the result in the strength map

## Architecture Notes

### CUDA Compute Capability
The Makefile is configured for compute capability 8.6 (`-arch=sm_86`). If you have a different GPU:

1. Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus
2. Update the Makefile's `CUDA_FLAGS` line:
   ```makefile
   CUDA_FLAGS = -O2 -arch=sm_XX  # Replace XX with your compute capability
   ```

Common compute capabilities:
- `sm_75`: RTX 20 series (Turing)
- `sm_86`: RTX 30 series (Ampere)
- `sm_89`: RTX 40 series (Ada Lovelace)

### Memory Management
The GPU implementation:
- Allocates device memory for gradient images and strength map
- Transfers input data to GPU (Host → Device)
- Executes the CUDA kernel
- Transfers results back to CPU (Device → Host)
- Properly frees all GPU memory

## Performance Considerations

### Expected Speedup
GPU acceleration is most beneficial for:
- Large images (>640x480)
- Large processing windows
- Multiple consecutive frames
- Batch processing

For small images, CPU overhead might outweigh GPU benefits.

### Optimization Tips
1. **Batch Processing**: Process multiple frames together to amortize transfer costs
2. **Stream Processing**: Use CUDA streams for overlapping computation and transfer
3. **Persistent Memory**: Keep data on GPU across frames when processing video
4. **Kernel Tuning**: Adjust thread block size based on your GPU architecture

## Troubleshooting

### CUDA Compiler Not Found
```
make: nvcc: Command not found
```
**Solution:** Install NVIDIA CUDA Toolkit and ensure `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA Runtime Error
```
CUDA Error: invalid device ordinal (line XX)
```
**Solution:** Ensure you have a CUDA-capable GPU and correct drivers installed:
```bash
nvidia-smi  # Check if GPU is detected
```

### Architecture Mismatch
```
error: compute capability X.X is required
```
**Solution:** Update `CUDA_FLAGS` in Makefile to match your GPU's compute capability.

### OpenCV Not Found
```
error: opencv_core: No such file or directory
```
**Solution:** Install OpenCV development libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# Or specify custom path in Makefile
LIB = -L/path/to/opencv/lib ...
```

## CPU-Only Build

If you need to build without GPU support, you can still use the original CPU version by checking out an earlier commit before the GPU changes were merged.

## Performance Comparison

To compare CPU vs GPU performance:
1. Build both CPU and GPU versions
2. Run with timing enabled
3. Compare execution times reported in console output

The GPU version prints:
```
Launching GPU kernel for corner strength estimation...
GPU execution finished in X.XXX ms
```

## Contributing

When modifying GPU code:
1. Test on multiple GPU architectures if possible
2. Add error checking for all CUDA calls
3. Profile performance before and after changes
4. Update documentation for any API changes

## References

- Original KLT Algorithm: http://www.ces.clemson.edu/~stb/klt
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA Profiling Tools: https://developer.nvidia.com/tools-overview

## Credits

GPU acceleration implemented by GulsherKhan-04:
- PR #2: Makefile updates for GPU/CPU compilation
- PR #3: CUDA implementation (selectGoodFeatures.cu)
- PR #4: Header file for GPU functions
