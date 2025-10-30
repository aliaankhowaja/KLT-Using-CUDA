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
#define BLOCKDIM 32
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 32
#define BLOCKDIM_HALO 86
#define DIMX blockDim.x
#define DIMY blockDim.y


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

__constant__ float horizKernelData[MAX_KERNEL_WIDTH];
__constant__ float vertKernelData[MAX_KERNEL_WIDTH];

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

__global__ void _convolveImageHorizGPU(float *imgin, float *imgout, int nrows, int ncols, int kernelWidth){
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
        sum += imgin[p] * horizKernelData[i];
      }
      imgout[idx] = sum;
    }
  }
}

/*********************************************************************
 * _convolveImageVert
 */

__global__ void _convolveImageVertGPU(float *imgin, float *imgout, int nrows, int ncols, int kernelWidth){
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
        sum += imgin[p] * vertKernelData[i];
      }
      imgout[idx] = sum;
    }
  }
}


  
/*********************************************************************
 * _convolveImageHoriz
 */

__global__ void _convolveImageHorizGPU_shared_mem1(float *imgin, float *imgout, int nrows, int ncols, int kernelWidth)
{
  int r = kernelWidth / 2;
  __shared__ float tile[BLOCKDIM][BLOCKDIM_HALO]; // shared memory tile 

  // global indices for convolution operation at each pixel
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // local indices block indices for shared membory
  int lrow = threadIdx.y;
  int lcol = threadIdx.x;

  // copy the data into shared memory
  if (row < nrows && col < ncols)
  {
    tile[lrow][lcol + r] = imgin[row * ncols + col]; // copy pixel data
    if (lcol == 0 && col >= r) // first thread loads left halos
    {
      for (int i = r; i > 0; i--) 
      {
        tile[lrow][r - i] = imgin[row * ncols + col - i];
      }
    }
    if (lcol == blockDim.x - 1 && col < ncols - r) // last thread loads right halos
    {
      for (int i = 1; i <= r; i ++){
        tile[lrow][lcol + r + i] = imgin[row * ncols + col + i];
      }
    }
  }

  __syncthreads();

  if (row < nrows && col < ncols)
  {
    int idx = row * ncols + col;
    if (col < r || col >= ncols - r)
    {
      imgout[idx] = 0;
    }
    else
    {
      float sum = 0;
      // Convolve using shared memory
      for (int i = kernelWidth - 1, p = lcol; i >= 0; i--, p++)
      {
        sum += tile[lrow][p] * horizKernelData[i];
      }

      imgout[idx] = sum;
    }
  }
}

/*********************************************************************
 * _convolveImageVert
 */

__global__ void _convolveImageVertGPU_shared_mem1(float *imgin, float *imgout, int nrows, int ncols, int kernelWidth){
  int r = kernelWidth / 2;
  __shared__ float tile[BLOCKDIM_HALO][BLOCKDIM]; // shared memory tile 
  
  // global indices for convolution operation at each pixel
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // local indices block indices for shared membory
  int lrow = threadIdx.y;
  int lcol = threadIdx.x;
  
  // copy the data into shared memory
  if (row < nrows && col < ncols)
  {
    tile[lrow+r][lcol] = imgin[row * ncols + col]; // copy pixel data
    if (lrow == 0 && row >= r) // first thread loads top halos
    {
      for (int i = r; i > 0; i--) 
      {
        tile[r - i][lcol] = imgin[(row - i) * ncols + col];
      }
    }
    if (lrow == blockDim.y - 1 && row < nrows - r) // last thread loads bottom halos
    {
      for (int i = 1; i <= r; i ++){
        tile[lrow + r + i][lcol] = imgin[(row + i) * ncols + col];
      }
    }
  }

  __syncthreads();

  if (row < nrows && col < ncols)
  {
    int idx = row * ncols + col;
    if (row < r || row >= nrows-r){
      imgout[idx] = 0;
    }
    else {
      float sum = 0;
      // Convolve using shared memory
      for (int i = kernelWidth-1, p = lrow; i >= 0; i--, p++)
      {
        sum += tile[p][lcol] * vertKernelData[i];
      }
      imgout[idx] = sum;
    }
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
  // _KLT_FloatImage tmpimg;
  // tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  int nrows = imgin->nrows;
  int ncols = imgin->ncols;

  float* imgin_d, * tmpimg_d, * imgout_d;
  int imgSize = nrows * ncols * sizeof(float);
  cudaMalloc((void**)&imgin_d, imgSize * 2);
  tmpimg_d = imgin_d + nrows * ncols;
  imgout_d = imgin_d; // reuse input memory for output

  cudaMemcpy(imgin_d, imgin->data, imgSize, cudaMemcpyHostToDevice);

  dim3 gridSize((ncols + BLOCKDIM_X - 1) / BLOCKDIM_X, (nrows + BLOCKDIM_Y - 1) / BLOCKDIM_Y);
  dim3 blockSize(BLOCKDIM_X, BLOCKDIM_Y);

  int horizKernelWidth = horiz_kernel.width;
  int vertKernelWidth = vert_kernel.width;
  cudaMemcpyToSymbol(horizKernelData, horiz_kernel.data, horizKernelWidth * sizeof(float));
  cudaMemcpyToSymbol(vertKernelData, vert_kernel.data, vertKernelWidth * sizeof(float));

  _convolveImageHorizGPU<<<gridSize, blockSize>>>(imgin_d, tmpimg_d, nrows, ncols, horizKernelWidth);
  cudaDeviceSynchronize();
  _convolveImageVertGPU<<<gridSize, blockSize>>>(tmpimg_d, imgout_d, nrows, ncols, vertKernelWidth);
  cudaDeviceSynchronize();

  cudaMemcpy(imgout->data, imgout_d, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(imgin_d);

  // /* Do convolution */
  // _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  // _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  // _KLTFreeFloatImage(tmpimg);
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



