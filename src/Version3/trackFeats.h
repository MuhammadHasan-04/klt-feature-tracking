#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "klt.h"  // for _KLT_FloatImage

/**
 * @brief Bilinear interpolation of a single point in a float image.
 *
 * This function returns the interpolated intensity value at coordinates (x, y)
 * from the given _KLT_FloatImage.
 *
 * @param x   X-coordinate (can be fractional)
 * @param y   Y-coordinate (can be fractional)
 * @param img Pointer to _KLT_FloatImage
 * @return    Interpolated float intensity value
 */
float _interpolate(float x, float y, _KLT_FloatImage img);

#ifdef __cplusplus
}
#endif

#endif // INTERPOLATE_H
