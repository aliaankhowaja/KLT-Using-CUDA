#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gpu/trackFeatures.cu"

void fillRandomGradients(_KLT_FloatImage img, float min_val, float max_val)
{
  int total = img->ncols * img->nrows;

  for (int i = 0; i < total; i++) 
  {
    float range = max_val - min_val;
    img->data[i] = min_val + ((float)rand() / RAND_MAX) * range;
  }
}

int compareResults(float *arr1, float *arr2, int size, float tolerance)
{
  int errors = 0;
  float max_diff = 0.0f;
  
  for (int i = 0; i < size; i++) 
  {
    float diff = fabs(arr1[i] - arr2[i]);

    if (diff > max_diff) 
        max_diff = diff;
    
    if (diff > tolerance) 
    {
      if (errors < 5) 
      {  /* Print first 5 errors only */
        printf("  Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n", i, arr1[i], arr2[i], diff);
      }
      errors++;
    }
  }
  
  printf("  Max difference: %.8f\n", max_diff);
  return errors;
}

int main()
{
  const int IMG_WIDTH = 50;
  const int IMG_HEIGHT = 50;
  const int WINDOW_WIDTH = 7;
  const int WINDOW_HEIGHT = 7;
  const float x1 = 25.3f;
  const float y1 = 25.7f;
  const float x2 = 26.1f;
  const float y2 = 24.9f;
  const float TOLERANCE = 1e-4f;
  
  printf("=== Testing GPU Gradient Sum Kernel ===\n\n");
  printf("Image size: %dx%d\n", IMG_WIDTH, IMG_HEIGHT);
  printf("Window size: %dx%d\n", WINDOW_WIDTH, WINDOW_HEIGHT);
  printf("Window center 1: (%.1f, %.1f)\n", x1, y1);
  printf("Window center 2: (%.1f, %.1f)\n", x2, y2);
  printf("Tolerance: %.6f\n\n", TOLERANCE);
  
  srand(42);
  
  _KLT_FloatImage gradx1 = _KLTCreateFloatImage(IMG_WIDTH, IMG_HEIGHT);
  _KLT_FloatImage grady1 = _KLTCreateFloatImage(IMG_WIDTH, IMG_HEIGHT);
  _KLT_FloatImage gradx2 = _KLTCreateFloatImage(IMG_WIDTH, IMG_HEIGHT);
  _KLT_FloatImage grady2 = _KLTCreateFloatImage(IMG_WIDTH, IMG_HEIGHT);
  
  printf("Filling gradient images with random values...\n");
  fillRandomGradients(gradx1, -50.0f, 50.0f);
  fillRandomGradients(grady1, -50.0f, 50.0f);
  fillRandomGradients(gradx2, -50.0f, 50.0f);
  fillRandomGradients(grady2, -50.0f, 50.0f);
  
  int window_size = WINDOW_WIDTH * WINDOW_HEIGHT;
  float *gradx_cpu = (float*)malloc(window_size * sizeof(float));
  float *grady_cpu = (float*)malloc(window_size * sizeof(float));
  float *gradx_gpu = (float*)malloc(window_size * sizeof(float));
  float *grady_gpu = (float*)malloc(window_size * sizeof(float));
  
  printf("\nRunning CPU version...\n");

  _computeGradientSumCPU(gradx1, grady1, gradx2, grady2,
                         x1, y1, x2, y2,
                         WINDOW_WIDTH, WINDOW_HEIGHT,
                         gradx_cpu, grady_cpu);

  printf("CPU completed.\n");
  
  printf("Running GPU version...\n");

  _computeGradientSum(gradx1, grady1, gradx2, grady2,
                      x1, y1, x2, y2,
                      WINDOW_WIDTH, WINDOW_HEIGHT,
                      gradx_gpu, grady_gpu);

  printf("GPU completed.\n");
  
  printf("\nFirst 5 output values:\n");

  printf("Index | CPU gradx  | GPU gradx  | CPU grady  | GPU grady\n");
  printf("------|------------|------------|------------|------------\n");

  for (int i = 0; i < 5 && i < window_size; i++) 
  {
    printf("  %2d  | %10.4f | %10.4f | %10.4f | %10.4f\n",
           i, gradx_cpu[i], gradx_gpu[i], grady_cpu[i], grady_gpu[i]);
  }
  
  printf("\n=== Comparing Results ===\n");
  
  printf("\nChecking gradx output:\n");

  int errors_x = compareResults(gradx_cpu, gradx_gpu, window_size, TOLERANCE);
  
  printf("\nChecking grady output:\n");

  int errors_y = compareResults(grady_cpu, grady_gpu, window_size, TOLERANCE);
  
  printf("\n=== Test Summary ===\n");
  printf("Total pixels compared: %d\n", window_size);
  printf("Errors in gradx: %d\n", errors_x);
  printf("Errors in grady: %d\n", errors_y);
  
  if (errors_x == 0 && errors_y == 0) 
  {
    printf("\n✓ TEST PASSED: GPU and CPU results match!\n");
  } 
  else 
  {
    printf("\n✗ TEST FAILED: GPU and CPU results differ!\n");
  }
  
  _KLTFreeFloatImage(gradx1);
  _KLTFreeFloatImage(grady1);
  _KLTFreeFloatImage(gradx2);
  _KLTFreeFloatImage(grady2);
  free(gradx_cpu);
  free(grady_cpu);
  free(gradx_gpu);
  free(grady_gpu);
  
  return (errors_x + errors_y > 0) ? 1 : 0;
}

/*********************************************************************
 * COMPILATION INSTRUCTIONS:
 * 
 * From the v2 directory, compile with:
 * 
 * nvcc -o test_gradient_sum test_gradient_sum.cu \
 *      -I./cpu -I./gpu -arch=compute_86 -code=sm_86
 * 
 * Then run with:
 * ./test_gradient_sum
 * 
 * IMPORTANT: Make sure use_gpu = TRUE in trackFeatures.cu (line 217)
 *            to actually test the GPU kernel!
 * 
 *********************************************************************/
