//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#include "device_functions.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#define NUM_BITS 4
#define RADIX (1 << NUM_BITS)
#define BLOCK_SIZE 256 // no more than 254 in order to fit all relative positions into the array of bytes

__device__ unsigned int get_digit(unsigned int num, unsigned int shift)
{
  int mask = (NUM_BITS - 1) << shift;
  return (num & mask) >> shift;
}

__global__ void hist_prelim(const float* const in, const size_t size, unsigned int shift, const int nBins, unsigned int* const  outBins)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
  {
    return;
  }

  unsigned int digit = get_digit(in[idx], shift);

  atomicAdd(&outBins[digit], 1);
}

// exclusive scan sum of the histograms of digits
__global__ void scan_sum_exclusive(const int * const in, const size_t size, unsigned int * const out, int * const sums)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ int tempIn[];

  tempIn[threadIdx.x] = threadIdx.x > 0 ? in[idx - 1] : 0;
  __syncthreads();

  for(int step = 1; step < size; step <<= 1)
  {
    __syncthreads();
    int tSum = tempIn[threadIdx.x];

    if(threadIdx.x >= step)
    {
      tSum += tempIn[threadIdx.x - step];
    }
    __syncthreads();
    tempIn[threadIdx.x] = tSum;
    
  }

  __syncthreads();
  out[idx] = tempIn[threadIdx.x];
  
  __syncthreads();

  int last = blockDim.x * blockIdx.x + size - 1;

  if(threadIdx.x == 0)
  {
    sums[blockIdx.x] = out[last] + in[last];
  }
}

// adding up partial sums to get the final output scan
__global__ void addSums(unsigned int * const inOut, int * const vals)
{
  if(blockIdx.x == 0)
  {
    return;
  }

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  inOut[idx] += vals[blockIdx.x - 1];
}

// Build a relative positional scan of digit, shifted by "shift"
// in - array of numbers to sort
// inPos - positions of the keys that also need to move
// size - size of the array
// shift - which significant digit are we concerned about
// rel - array of relative positions of the numbers in the in array (relative to current block), based on the digit of interest.
__global__ void rel_pos_per_block(const unsigned int * const in, 
                                     const size_t size, unsigned char shift, 
                                     unsigned char * const rel)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int mask = (NUM_BITS - 1) << shift;
  int digit_pos = (in[idx] & mask) >> shift;

  int offset = digit_pos * BLOCK_SIZE;

  __shared__ unsigned char relativePosition[RADIX * BLOCK_SIZE];
  for(int i = 0; i < RADIX; i++)
  {
    relativePosition[threadIdx.x + i * BLOCK_SIZE] = 0;
  }
  __syncthreads();

  // initialize to the "startup" positions for each digit.
  relativePosition[threadIdx.x + digit_pos * BLOCK_SIZE] = threadIdx.x == 0 ? 0 : 1;

  // accumulate position values for each digit
  for(int step = 1; step < size; step <<= 1)
  {
    __syncthreads();

    // position start of the digit in the relatvePosition array
    int tSum = relativePosition[threadIdx.x + offset];

    if(threadIdx.x >= step)
    {
      tSum += relativePosition[threadIdx.x + offset - step];
    }
    __syncthreads();
    relativePosition[threadIdx.x + offset] = tSum;
  }

  __syncthreads();
  //combine relative positions with "start" positions

  rel[idx] = relativePosition[threadIdx.x + offset];
  __syncthreads();
}

// Convert relative postions to absolute ones by adding up to the histogram
// rel array of relative positions
// cdf - exclusive cdf of the counts histogram
__global__ void move_to_out(unsigned int * const dest, 
                            unsigned int * const destPos, 
                            unsigned int * const source, 
                            unsigned int * const sourcePos, 
                            unsigned int * const rel, 
                            size_t size, 
                            unsigned int shift, 
                            unsigned int * const absPos, 
                            unsigned int * const cdf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size)
  {
    return;
  }

  int pos = cdf[get_digit(source[idx], shift)] + rel[idx];
  dest[pos] = source[idx];
  destPos[pos] = sourcePos[idx];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
}
