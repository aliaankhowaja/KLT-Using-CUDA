/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */
#include <cuda.h>

#define MAX_KERNEL_WIDTH 	71


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}


/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

__global__ void _convolveImageHorizGPU(float *imgin, float *imgout, float* kernel, int nrows, int ncols, int kernelWidth){
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int r = kernelWidth / 2;
  if (row < nrows && col < ncols)
  {
    int idx = row * ncols + col;
    if (col < r || col >= ncols-r){
      imgout[idx] = 0;
    }
    else {
      float sum = 0;
      for (int i = kernelWidth-1, p = idx - r; i >= 0; i--, p++)
      {
        sum += imgin[p] * kernel[i];
      }
      imgout[idx] = sum;
    }
  }
}

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  // float *ptrrow = imgin->data;           /* Points to row's first pixel */
  // register
  //  float *ptrout = imgout->data, /* Points to next output pixel */
    // *ppp;
  // register
  //  float sum;
  // register
   int radius = kernel.width / 2;
  // register
   int ncols = imgin->ncols, nrows = imgin->nrows;
  // register
   int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  float *kernel_d; // kernel
  float *img_d; // input and output images
  int inSize = nrows * ncols * sizeof(float);
  int outSize = imgout->nrows * imgout->ncols * sizeof(float);
  int kernelSize = kernel.width * sizeof(float);
  cudaMalloc((void **)&img_d, inSize + outSize);
  cudaMalloc((void **)&kernel_d, kernelSize);
  cudaMemcpy(kernel_d, kernel.data, kernelSize, cudaMemcpyHostToDevice);
  float *imgout_d = img_d + nrows*ncols;
  cudaMemcpy(img_d, imgin->data, inSize, cudaMemcpyHostToDevice);

  dim3 gridSize((ncols + 7) / 8, (nrows + 7) / 8);
  dim3 blockSize(8, 8);

  _convolveImageHorizGPU<<<gridSize, blockSize>>>(img_d, imgout_d, kernel_d, nrows, ncols, kernel.width);
  cudaDeviceSynchronize();

  cudaMemcpy(imgout->data, imgout_d, outSize, cudaMemcpyDeviceToHost);
  cudaFree(img_d);
  cudaFree(kernel_d);
}

static void _convolveImageHorizCPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;           /* Points to row's first pixel */
  // 
  float *ptrout = imgout->data, /* Points to next output pixel */
    *ppp;
  // 
  float sum;
  // 
  int radius = kernel.width / 2;
  // 
  int ncols = imgin->ncols, nrows = imgin->nrows;
  // 
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  for (j = 0 ; j < nrows ; j++)  {

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}


/*********************************************************************
 * _convolveImageVert
 */

__global__ void _convolveImageVertGPU(float *imgin, float *imgout, float* kernel, int nrows, int ncols, int kernelWidth){
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int r = kernelWidth / 2;
  if (row < nrows && col < ncols)
  {
    int idx = row * ncols + col;
    if (row < r || row >= nrows-r){
      imgout[idx] = 0;
    }
    else {
      float sum = 0;
      for (int i = kernelWidth-1, p = idx - r*ncols; i >= 0; i--, p+=ncols)
      {
        sum += imgin[p] * kernel[i];
      }
      imgout[idx] = sum;
    }
  }
}

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  // float *ptrcol = imgin->data;            /* Points to row's first pixel */
  // float *ptrout = imgout->data,  /* Points to next output pixel */
    // *ppp;
  // float sum;
  // 
  int radius = kernel.width / 2;
  // 
  int ncols = imgin->ncols, nrows = imgin->nrows;
  // 
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  float *kernel_d; // kernel
  float *img_d; // input and output images
  int inSize = nrows * ncols * sizeof(float);
  int outSize = imgout->nrows * imgout->ncols * sizeof(float);
  int kernelSize = kernel.width * sizeof(float);
  cudaMalloc((void **)&img_d, inSize + outSize);
  cudaMalloc((void **)&kernel_d, kernelSize);
  cudaMemcpy(kernel_d, kernel.data, kernelSize, cudaMemcpyHostToDevice);
  float *imgout_d = img_d + nrows*ncols;
  cudaMemcpy(img_d, imgin->data, inSize, cudaMemcpyHostToDevice);

  dim3 gridSize((ncols + 7) / 8, (nrows + 7) / 8);
  dim3 blockSize(8, 8);

  _convolveImageVertGPU<<<gridSize, blockSize>>>(img_d, imgout_d, kernel_d, nrows, ncols, kernel.width);
  cudaDeviceSynchronize();

  cudaMemcpy(imgout->data, imgout_d, outSize, cudaMemcpyDeviceToHost);
  cudaFree(img_d);
  cudaFree(kernel_d);
}

static void _convolveImageVertCPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;            /* Points to row's first pixel */
  // register
   float *ptrout = imgout->data,  /* Points to next output pixel */
    *ppp;
  // register
   float sum;
  // register
   int radius = kernel.width / 2;
  // register
   int ncols = imgin->ncols, nrows = imgin->nrows;
  // register
   int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}


/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}



