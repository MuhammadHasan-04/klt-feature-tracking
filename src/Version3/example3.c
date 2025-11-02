#include "pnmio.h"
#include "klt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 // #define FRAME_DIR "../../resized_frames/"  //for 1080p
#define FRAME_DIR "../../data/"  //for 320x240p
#define NUM_FRAMES 100
#define N_FEATURES 150   // Number of features to track per frame

int main() {
    clock_t start = clock();

    unsigned char *img1 = NULL, *img2 = NULL;
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    int ncols, nrows;

    // Create tracking context and feature list
    tc = KLTCreateTrackingContext();
    tc->sequentialMode = TRUE;  // optimize for sequential frames
    tc->writeInternalImages = FALSE;
    tc->affineConsistencyCheck = -1; // optional, can enable

    KLTPrintTrackingContext(tc);

    fl = KLTCreateFeatureList(N_FEATURES);

    char fname1[256], fname2[256], out_ppm[256], out_txt[256];

    // Load the first frame
    sprintf(fname1, "%s%010d.pgm", FRAME_DIR, 0);
    img1 = pgmReadFile(fname1, NULL, &ncols, &nrows);
    if (!img1) {
        printf("[ERROR] Could not open first frame %s\n", fname1);
        return -1;
    }

    // Select good features in the first frame
    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

    // Save initial feature visualization
    sprintf(out_ppm, "feat_%04d.ppm", 0);
    sprintf(out_txt, "feat_%04d.txt", 0);
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, out_ppm);
    KLTWriteFeatureList(fl, out_txt, "%3d");

    // Track features across remaining frames
    for (int f = 1; f < NUM_FRAMES; f++) {
        sprintf(fname2, "%s%010d.pgm", FRAME_DIR, f);
        printf("[INFO] Tracking %s -> %s\n", fname1, fname2);

        img2 = pgmReadFile(fname2, NULL, &ncols, &nrows);
        if (!img2) {
            printf("[ERROR] Could not open frame %d\n", f);
            break;
        }

        // Track features from img1 -> img2
        KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

        // Optionally replace lost features
        // KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);

        // Save visualization for this frame
        sprintf(out_ppm, "feat_%04d.ppm", f);
        sprintf(out_txt, "feat_%04d.txt", f);
        KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, out_ppm);
        KLTWriteFeatureList(fl, out_txt, "%3d");

        // Swap buffers for next iteration
        free(img1);
        img1 = img2;
        strcpy(fname1, fname2);
    }

    // Clean up
    KLTFreeFeatureList(fl);
    KLTFreeTrackingContext(tc);
    free(img1);

    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total (CPU + GPU) time for %d frames: %.6f seconds\n", NUM_FRAMES, cpu_time);

    return 0;
}
