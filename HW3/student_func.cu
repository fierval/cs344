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
#include "timer.h"
#include "device_functions.h"
#include <memory>

#pragma warning (disable: 4267)

const int BLOCK_SIZE = 512;

__global__ void extent(const float* const in, const size_t size, float* const outMin, float * const outMax)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float tempIn[BLOCK_SIZE];
  __shared__ float tempInMax[BLOCK_SIZE];

  tempIn[threadIdx.x] = in[idx];
  tempInMax[threadIdx.x] = tempIn[threadIdx.x];

  for(int step = 1; step < size; step <<= 1)
  {
    __syncthreads();
    if (threadIdx.x >= step)
    {
      float tMin = tempIn[threadIdx.x] < tempIn[threadIdx.x - step] ? tempIn[threadIdx.x] : tempIn[threadIdx.x - step];
      float tMax = tempInMax[threadIdx.x] > tempInMax[threadIdx.x - step] ? tempInMax[threadIdx.x] : tempInMax[threadIdx.x - step];  

      __syncthreads();

      tempInMax[threadIdx.x] = tMax;
      tempIn[threadIdx.x] = tMin;
    }
  }


  // last thread writes out the result
  if (threadIdx.x == size - 1)
  {
    __syncthreads();

    outMin[blockIdx.x] = tempIn[threadIdx.x];
    outMax[blockIdx.x] = tempInMax[threadIdx.x];
  }

}

__global__ void scan_sum_exclusive(const int * const in, const size_t size, unsigned int * const out, int * const sums)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ int tempIn[];

  tempIn[threadIdx.x] = threadIdx.x > 0 ? in[idx - 1] : 0;

  for(int step = 1; step < size; step <<= 1)
  {
    __syncthreads();
    if(threadIdx.x >= step)
    {
      int tSum = tempIn[threadIdx.x] + tempIn[threadIdx.x - step];
      __syncthreads();
      tempIn[threadIdx.x] = tSum;
    }
  }

  __syncthreads();
  out[idx] = tempIn[threadIdx.x];
  
  __syncthreads();
  if(threadIdx.x == 0)
  {
    int last = blockDim.x * blockIdx.x + size - 1;
    sums[blockIdx.x] = out[last] + in[last];
  }
}

__global__ void addSums(unsigned int * const inOut, int * const vals)
{
  if(blockIdx.x == 0)
  {
    return;
  }

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  inOut[idx] += vals[blockIdx.x - 1];
}

__global__ void hist_prelim(const float* const in, const size_t size, const int nBins, const float lumMin, const float lumRange, int* const  outBins)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
  {
    return;
  }

  int bin = (in[idx] - lumMin) / lumRange * nBins;
  if (bin > nBins - 1)
    bin = nBins - 1;

  atomicAdd(&outBins[bin], 1);
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
  dim3 blockSize(BLOCK_SIZE);
  dim3 gridSize(size / BLOCK_SIZE);

  int remainder = size % BLOCK_SIZE;

  std::unique_ptr<float> h_minOut(new float[gridSize.x]);
  std::unique_ptr<float> h_maxOut(new float[gridSize.x]);

  float * d_minOut, *d_maxOut;
  min_logLum = FLT_MAX;
  max_logLum = 0;

#ifdef _DEBUG
  GpuTimer timer;

  timer.Start();
#endif

  checkCudaErrors(cudaMalloc(&d_minOut, sizeof(float) * gridSize.x));
  checkCudaErrors(cudaMalloc(&d_maxOut, sizeof(float) * gridSize.x));

  extent<<<gridSize, blockSize>>>(d_logLuminance, BLOCK_SIZE, d_minOut, d_maxOut);

  if(remainder > 0)
  {
    extent<<<1, remainder>>>(&d_logLuminance[gridSize.x * blockSize.x], remainder, &d_minOut[gridSize.x], &d_maxOut[gridSize.x]);
  }

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

  float seq_min = std::numeric_limits<float>::max(), seq_max = -std::numeric_limits<float>::max();
  checkCudaErrors(cudaMemcpy(h_in, d_logLuminance, size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++)
  {
    seq_min = std::min(seq_min, *(h_in + i));
    seq_max = std::max(seq_max, *(h_in + i));
  }

  duration =  ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000.;

  printf("CUDA min\\max: %f, %f, elapsed: %lf\n", min_logLum, max_logLum, timer.Elapsed());
  printf("Seq min\\max: %f, %f, elapsed: %lf\n", seq_min, seq_max, duration);
  free(h_in);
#endif

  /*2) subtract them to find the range */
  float lumRange = max_logLum - min_logLum;

  /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
#ifdef _DEBUG
  timer.Start();
#endif


  int *d_bins;
  std::unique_ptr<int> hist(new int[numBins]);

  checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
  checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));

  hist_prelim<<<gridSize, blockSize>>>(d_logLuminance, size, numBins, min_logLum, lumRange, d_bins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(hist.get(), d_bins, sizeof(int) * numBins, cudaMemcpyDeviceToHost));

#ifdef _DEBUG
  timer.Stop();

  h_in = (float *) malloc(size * sizeof(float));
  start = std::clock();

  checkCudaErrors(cudaMemcpy(h_in, d_logLuminance, size * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned int *histo = new unsigned int[numBins];

  for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

  for (size_t i = 0; i < numCols * numRows; ++i) {
    unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((h_in[i] - min_logLum) / lumRange * numBins));
    histo[bin]++;
  }

  duration =  ( std::clock() - start ) / (double) CLOCKS_PER_SEC * 1000;

  printf("CUDA hist: %d, %d %d, elapsed: %lf\n", hist.get()[0], hist.get()[1], hist.get()[numBins - 1], timer.Elapsed());
  printf("Seq hist: %d, %d %d, elapsed: %lf\n", histo[0], histo[1], histo[numBins - 1], seq_max, duration);

  delete[] histo;
  free(h_in);

#endif
  /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       
  */
    
    gridSize.x = ((int)numBins / BLOCK_SIZE);
    const int remaining = (int)numBins % BLOCK_SIZE;

    std::unique_ptr<int> scanned(new int[numBins]);

    int *d_sums;

    checkCudaErrors(cudaMalloc(&d_sums, sizeof(int) * (gridSize.x + 1)));

    // compute main scan
    if (gridSize.x > 0)
    {
      scan_sum_exclusive<<<gridSize, blockSize, blockSize.x * sizeof(int)>>>(d_bins, blockSize.x, d_cdf, d_sums);
    }
      
    // launch one more block of the "remainder" elements
    if(remaining > 0)
    {
      scan_sum_exclusive<<<1, remaining, remaining * sizeof(int)>>>(&d_bins[gridSize.x * blockSize.x], remaining, &d_cdf[gridSize.x * blockSize.x], &d_sums[gridSize.x]);
    }

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    if (gridSize.x > 0)
    {
      std::unique_ptr<int> sums(new int[gridSize.x + 1]);
      checkCudaErrors(cudaMemcpy(sums.get(), d_sums, (gridSize.x + 1) * sizeof(int), cudaMemcpyDeviceToHost));

      for(unsigned int i = 1; i < gridSize.x + 1; i++)
      {
        sums.get()[i] += sums.get()[i - 1];
      }

      int sizeAdd = remaining > 0 ? 1 : 0;
      
      checkCudaErrors(cudaMemcpy(d_sums, sums.get(), (gridSize.x + 1) * sizeof(int), cudaMemcpyHostToDevice));

      addSums<<<gridSize.x + sizeAdd, blockSize>>>(d_cdf, d_sums);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaMemcpy(scanned.get(), d_cdf, numBins * sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_bins));
    checkCudaErrors(cudaFree(d_sums));
}
