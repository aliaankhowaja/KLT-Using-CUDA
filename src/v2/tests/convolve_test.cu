#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "../cpu/klt.h"
#include "../cpu/klt_util.h"
#include "../gpu/convolve.h"


// Function to compare two float images
void compare_images(_KLT_FloatImage img1, _KLT_FloatImage img2, float tolerance) {
    assert(img1->ncols == img2->ncols && img1->nrows == img2->nrows);
    for (int i = 0; i < img1->ncols * img1->nrows; ++i) {
        if (fabs(img1->data[i] - img2->data[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, img1->data[i], img2->data[i]);
            return;
        }
    }
    printf("Images are identical within the tolerance.\n");
}

int main() {
    const int ncols = 256;
    const int nrows = 256;
    // const float sigma = 1.0f;
    ConvolutionKernel kernel;
    kernel.width = 5;
    kernel.data = (float*)malloc(kernel.width * sizeof(float));
    kernel.data[0] = 0.1;
    kernel.data[1] = 0.2;
    kernel.data[2] = 0.3;
    kernel.data[3] = 0.4;
    kernel.data[4] = 0.5;

    // Allocate memory for input image
    KLT_PixelType* img_pixels = (KLT_PixelType*)malloc(sizeof(KLT_PixelType) * ncols * nrows);

    // Initialize with random data
    srand(time(NULL));
    for (int i = 0; i < ncols * nrows; ++i) {
        img_pixels[i] = rand() % 256;
    }

    // Create float image
    _KLT_FloatImage img = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img_pixels, ncols, nrows, img);

    // Create output images for CPU and GPU
    _KLT_FloatImage imgout_cpu = _KLTCreateFloatImage(ncols, nrows);
    _KLT_FloatImage imgout_gpu = _KLTCreateFloatImage(ncols, nrows);

    // Run CPU convolution
    _convolveImageHorizCPU(img, kernel, imgout_cpu);

    // Run GPU convolution
    _convolveImageHoriz(img, kernel, imgout_gpu);

    // Compare results
    printf("Comparing CPU and GPU horizontal convolution results...\n");
    compare_images(imgout_cpu, imgout_gpu, 1e-4);

    // Run CPU convolution
    _convolveImageHorizCPU(img, kernel, imgout_cpu);

    // Run GPU convolution
    _convolveImageHoriz(img, kernel, imgout_gpu);

    // Compare results
    printf("Comparing CPU and GPU vertical convolution results...\n");
    compare_images(imgout_cpu, imgout_gpu, 1e-4);

    // Free memory
    free(img_pixels);
    _KLTFreeFloatImage(img);
    _KLTFreeFloatImage(imgout_cpu);
    _KLTFreeFloatImage(imgout_gpu);

    return 0;
}
