/*********************************************************************
 * selectGoodFeatures.h
 * Header for GPU-accelerated KLT feature selection
 *********************************************************************/

#ifndef SELECT_GOOD_FEATURES_H
#define SELECT_GOOD_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "klt.h"

/**
 * @brief Launches the CUDA kernel to compute the feature strength map.
 *
 * This function transfers gradient images (gradx, grady) to the GPU,
 * runs the computeFeatureStrengthKernel to calculate the minimum eigenvalue
 * for each pixel, and copies the resulting strength map back to host memory.
 *
 * @param gradx       Input gradient image in X direction
 * @param grady       Input gradient image in Y direction
 * @param strength    Output strength image (trackability map)
 * @param win_width   Width of the processing window
 * @param win_height  Height of the processing window
 * @param borderx     Border in X to ignore (avoids out-of-bound memory)
 * @param bordery     Border in Y to ignore
 */
void gpuComputeFeatureStrength(
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady,
    _KLT_FloatImage strength,
    int win_width,
    int win_height,
    int borderx,
    int bordery
);

#ifdef __cplusplus
}
#endif

#endif /* SELECT_GOOD_FEATURES_H */


