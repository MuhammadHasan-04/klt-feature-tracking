#include "pnmio.h"
#include "klt.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 // #define FRAME_DIR "../../resized_frames/"  //for 1080p
#define FRAME_DIR "../../data/"  //for 320x240p
#define NUM_FRAMES 100

int main() {
    clock_t start = clock();

    unsigned char *img1 = NULL, *img2 = NULL;
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    int nFeatures = 100;
    int ncols, nrows;
    int i;

    tc = KLTCreateTrackingContext();
    KLTPrintTrackingContext(tc);
    fl = KLTCreateFeatureList(nFeatures);

    char fname1[256], fname2[256], out_ppm[256], out_txt[256];

    // Load first frame
    sprintf(fname1, "%s%010d.pgm", FRAME_DIR, 0);
    img1 = pgmReadFile(fname1, NULL, &ncols, &nrows);
    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

    // Write initial frame visualization
    sprintf(out_ppm, "feat_%04d.ppm", 0);
    sprintf(out_txt, "feat_%04d.txt", 0);
    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, out_ppm);
    KLTWriteFeatureList(fl, out_txt, "%3d");

    // Process remaining frames
    for (int f = 1; f < NUM_FRAMES; f++) {
        sprintf(fname2, "%s%010d.pgm", FRAME_DIR, f);
        printf("[INFO] Tracking %s -> %s\n", fname1, fname2);

        img2 = pgmReadFile(fname2, NULL, &ncols, &nrows);
        if (!img2) {
            printf("[ERROR] Could not open frame %d\n", f);
            break;
        }

        KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

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

    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total (CPU + GPU) time: %.6f seconds\n", cpu_time);

    return 0;
}
