/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "device_launch_parameters.h"
//#include "sm_11_atomic_functions.h"
#include "timer.h"
#include <memory>

const int BLOCK_SIZE = 512;

__global__ void extent(const float* const in, const size_t size, float* const outMin, float * const outMax, int * const sizeTracker)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
  {
    atomicAdd(sizeTracker, -1);
    return;
  }

  __shared__ float tempIn[BLOCK_SIZE];
  __shared__ float tempInMax[BLOCK_SIZE];

  tempIn[threadIdx.x] = in[idx];
  tempInMax[threadIdx.x] = tempIn[threadIdx.x];
  __syncthreads();

  for(int step = 1; step < *sizeTracker; step <<= 1)
  {
    if (threadIdx.x >= step)
    {
      tempIn[threadIdx.x] = min(tempIn[threadIdx.x], tempIn[threadIdx.x - step]);
      tempInMax[threadIdx.x] = max(tempInMax[threadIdx.x], tempInMax[threadIdx.x - step]);  
    }
  }

  __syncthreads();

  // last thread writes out the result
  if (threadIdx.x == *sizeTracker - 1)
  {
    outMin[blockIdx.x] = tempIn[threadIdx.x];
    outMax[blockIdx.x] = tempInMax[threadIdx.x];
  }

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum */


  const size_t size = numCols * numRows;
  const dim3 blockSize(BLOCK_SIZE);
  const dim3 gridSize((size + BLOCK_SIZE) / BLOCK_SIZE);

  std::unique_ptr<float> h_minOut(new float[gridSize.x]);
  std::unique_ptr<float> h_maxOut(new float[gridSize.x]);

  float * d_minOut, *d_maxOut;
  int *d_sizeTracker;
  min_logLum = FLT_MAX;
  max_logLum = 0;

#ifdef _DEBUG
  GpuTimer timer;

  timer.Start();
#endif

  checkCudaErrors(cudaMalloc(&d_minOut, sizeof(float) * gridSize.x));
  checkCudaErrors(cudaMalloc(&d_maxOut, sizeof(float) * gridSize.x));
  checkCudaErrors(cudaMalloc(&d_sizeTracker, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_sizeTracker, &BLOCK_SIZE, sizeof(int), cudaMemcpyHostToDevice));

  extent<<<gridSize, blockSize>>>(d_logLuminance, size, d_minOut, d_maxOut, d_sizeTracker);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_minOut.get(), d_minOut, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_maxOut.get(), d_maxOut, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost));

  //copy from device and find actual min/max
  for (int i = 0; i < (int) gridSize.x; i++)
  {
    min_logLum = std::min(min_logLum, h_minOut.get()[i]);
    max_logLum = std::max(max_logLum, h_maxOut.get()[i]);
  }

  checkCudaErrors(cudaFree(d_maxOut));
  checkCudaErrors(cudaFree(d_minOut));
  checkCudaErrors(cudaFree(d_sizeTracker));

#ifdef _DEBUG
  timer.Stop();
#endif // _DEBUG

// check agains sequential,
// compare perf
#ifdef _DEBUG
  float * h_in = (float *) malloc(size * sizeof(float));
  std::clock_t start;
  double duration;
  start = std::clock();

  float seq_min = FLT_MAX, seq_max = 0;
  checkCudaErrors(cudaMemcpy(h_in, d_logLuminance, size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++)
  {
    seq_min = std::min(seq_min, *(h_in + i));
    seq_max = std::max(seq_max, *(h_in + i));
  }

  duration =  ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

  printf("CUDA min\\max: %f, %f, elapsed: %lf\n", min_logLum, max_logLum, timer.Elapsed() / 1000.);
  printf("Seq min\\max: %f, %f, elapsed: %lf\n", seq_min, seq_max, duration);
  free(h_in);
#endif

  /*2) subtract them to find the range */
  float lumRange = max_logLum - min_logLum;

    /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins

    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       
  */
}
