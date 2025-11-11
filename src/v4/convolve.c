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
 * _convolveImageHoriz  with openacc
 */


static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
    float *data_in = imgin->data;
    float *data_out = imgout->data;
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int radius = kernel.width / 2;
    int kernel_width = kernel.width;

    assert(kernel.width % 2 == 1);
    assert(imgin != imgout);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    /* Copy kernel data to device constant-like memory */
    float kernel_data[MAX_KERNEL_WIDTH];
    for (int i = 0; i < kernel_width; i++) {
        kernel_data[i] = kernel.data[kernel_width - 1 - i]; /* Pre-reverse for better memory access */
    }

    /* Optimized OpenACC implementation similar to CUDA pattern */
    #pragma acc data copyin(data_in[0:nrows*ncols], kernel_data[0:kernel_width]) \
                    copyout(data_out[0:nrows*ncols])
    {
        #pragma acc parallel loop gang vector collapse(2) independent
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                int idx = row * ncols + col;
                
                /* Boundary handling - same as CUDA */
                if (col < radius || col >= ncols - radius) {
                    data_out[idx] = 0.0f;
                } else {
                    /* Optimized memory access pattern - similar to CUDA */
                    float sum = 0.0f;
                    int start_p = idx - radius;
                    
                    #pragma acc loop seq
                    for (int k = 0; k < kernel_width; k++) {
                        sum += data_in[start_p + k] * kernel_data[k];
                    }
                    data_out[idx] = sum;
                }
            }
        }
    }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
    float *data_in = imgin->data;
    float *data_out = imgout->data;
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int kernel_width = kernel.width;

    // Basic parallelization only
    #pragma acc parallel loop copyin(data_in[0:nrows*ncols], kernel) copyout(data_out[0:nrows*ncols])
    for (int col = 0; col < ncols; col++) {
        for (int row = 0; row < nrows; row++) {
            int radius = kernel_width / 2;
            if (row < radius || row >= nrows - radius) {
                data_out[row * ncols + col] = 0.0f;
                continue;
            }
            
            float sum = 0.0f;
            for (int k = 0; k < kernel_width; k++) {
                int p_row = row - radius + k;
                // Inefficient memory access pattern
                sum += data_in[p_row * ncols + col] * kernel.data[k];
            }
            data_out[row * ncols + col] = sum;
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



