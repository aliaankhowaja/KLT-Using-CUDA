/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 * _KLTComputePyramid - OpenACC Optimized Version
 */

void _KLTComputePyramid(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i, x, y;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* OpenACC: Create data region for entire pyramid computation */
  #pragma acc enter data copyin(img->data[0:img->ncols*img->nrows])
  
  /* Copy original image to level 0 of pyramid - parallelized */
  #pragma acc parallel loop present(img->data[0:img->ncols*img->nrows]) \
                          copyout(pyramid->img[0]->data[0:pyramid->ncols[0]*pyramid->nrows[0]])
  for (int idx = 0; idx < pyramid->ncols[0] * pyramid->nrows[0]; idx++) {
    pyramid->img[0]->data[idx] = img->data[idx];
  }

  /* Process each pyramid level */
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    
    /* Compute smoothed image - assuming _KLTComputeSmoothedImage is OpenACC optimized */
    _KLTComputeSmoothedImage(pyramid->img[i-1], sigma, tmpimg);

    /* Subsample with OpenACC parallelization */
    oldncols = ncols;
    ncols /= subsampling;  
    nrows /= subsampling;
    
    #pragma acc parallel loop collapse(2) present(tmpimg->data[0:oldncols*nrows*subsampling]) \
                                        copyout(pyramid->img[i]->data[0:ncols*nrows])
    for (y = 0 ; y < nrows ; y++) {
      for (x = 0 ; x < ncols ; x++) {
        int src_idx = (subsampling * y + subhalf) * oldncols + (subsampling * x + subhalf);
        int dst_idx = y * ncols + x;
        pyramid->img[i]->data[dst_idx] = tmpimg->data[src_idx];
      }
    }

    /* Update sigma for next level */
    sigma *= subsampling;
    
    _KLTFreeFloatImage(tmpimg);
  }
  
  /* Clean up data region */
  #pragma acc exit data delete(img->data[0:img->ncols*img->nrows])
}


/*********************************************************************
 * _KLTComputePyramidFast - Highly Optimized OpenACC Version
 * Uses combined operations and minimizes temporary memory
 */

void _KLTComputePyramidFast(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  int i;
  
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramidFast)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Single data region for entire pyramid computation */
 // #pragma acc data copyin(img->data[0:img->ncols*img->nrows]) \
                   copyout(pyramid->img[0:pyramid->nLevels]->data[0:...])
  {
    /* Level 0: Direct copy from input */
    int level0_size = pyramid->ncols[0] * pyramid->nrows[0];
    #pragma acc parallel loop present(img, pyramid->img[0])
    for (int idx = 0; idx < level0_size; idx++) {
      pyramid->img[0]->data[idx] = img->data[idx];
    }

    /* Process subsequent levels */
    for (i = 1; i < pyramid->nLevels; i++) {
      float sigma = i * subsampling * sigma_fact;
      int old_cols = pyramid->ncols[i-1];
      int old_rows = pyramid->nrows[i-1];
      int new_cols = pyramid->ncols[i];
      int new_rows = pyramid->nrows[i];
      
      /* Combined smoothing and subsampling in single kernel */
      #pragma acc parallel loop collapse(2) present(pyramid->img[i-1], pyramid->img[i])
      for (int y = 0; y < new_rows; y++) {
        for (int x = 0; x < new_cols; x++) {
          /* Simple box filter approximation for faster computation */
          float sum = 0.0f;
          int count = 0;
          
          /* Small neighborhood sampling instead of full convolution */
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              int src_x = subsampling * x + subhalf + dx;
              int src_y = subsampling * y + subhalf + dy;
              
              if (src_x >= 0 && src_x < old_cols && src_y >= 0 && src_y < old_rows) {
                sum += pyramid->img[i-1]->data[src_y * old_cols + src_x];
                count++;
              }
            }
          }
          
          pyramid->img[i]->data[y * new_cols + x] = (count > 0) ? sum / count : 0.0f;
        }
      }
    }
  }
}


/*********************************************************************
 * _KLTComputePyramidSimple - Basic OpenACC Version
 * For when you want to make incremental optimization commits
 */

void _KLTComputePyramidSimple(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;
  int oldncols;
  int i, x, y;
	
  /* Basic OpenACC - just parallelize the subsampling loops */
  
  /* Copy level 0 */
  memcpy(pyramid->img[0]->data, img->data, ncols*nrows*sizeof(float));

  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    _KLTComputeSmoothedImage(pyramid->img[i-1], sigma, tmpimg);

    /* Only this part gets OpenACC */
    oldncols = ncols;
    ncols /= subsampling;  
    nrows /= subsampling;
    
    #pragma acc parallel loop copyin(tmpimg->data[0:oldncols*nrows*subsampling]) \
                            copyout(pyramid->img[i]->data[0:ncols*nrows])
    for (y = 0 ; y < nrows ; y++) {
      for (x = 0 ; x < ncols ; x++) {
        pyramid->img[i]->data[y*ncols+x] = 
          tmpimg->data[(subsampling*y+subhalf)*oldncols + (subsampling*x+subhalf)];
      }
    }

    sigma *= subsampling;
    _KLTFreeFloatImage(tmpimg);
  }
}


/*********************************************************************
 * _KLTGetPyramidLevel - Utility function with OpenACC
 */

_KLT_FloatImage _KLTGetPyramidLevel(
  _KLT_Pyramid pyramid,
  int level)
{
  if (level < 0 || level >= pyramid->nLevels) {
    KLTError("(_KLTGetPyramidLevel) Invalid pyramid level %d", level);
    return NULL;
  }
  
  /* Ensure data is on host if needed */
  #pragma acc update self(pyramid->img[level]->data[0:pyramid->ncols[level]*pyramid->nrows[level]]) \
                   if_present
  
  return pyramid->img[level];
}


/*********************************************************************
 * _KLTSyncPyramidToHost - Force synchronization of pyramid to host
 */

void _KLTSyncPyramidToHost(_KLT_Pyramid pyramid)
{
  int i;
  
  #pragma acc data present(pyramid->img[0:pyramid->nLevels])
  {
    for (i = 0; i < pyramid->nLevels; i++) {
      int size = pyramid->ncols[i] * pyramid->nrows[i];
      #pragma acc update self(pyramid->img[i]->data[0:size])
    }
  }
}


/*********************************************************************
 * _KLTSyncPyramidToDevice - Force synchronization of pyramid to device
 */

void _KLTSyncPyramidToDevice(_KLT_Pyramid pyramid)
{
  int i;
  
  #pragma acc data present(pyramid->img[0:pyramid->nLevels])
  {
    for (i = 0; i < pyramid->nLevels; i++) {
      int size = pyramid->ncols[i] * pyramid->nrows[i];
      #pragma acc update device(pyramid->img[i]->data[0:size])
    }
  }
}