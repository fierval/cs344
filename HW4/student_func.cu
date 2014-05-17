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

// Build a relative positional scan of digit, shifted by "shift"
// in - array of numbers to sort
// inPos - positions of the keys that also need to move
// size - size of the array
// shift - which significant digit are we concerned about
// rel - array of relative positions of the numbers in the in array (relative to current block), based on the digit of interest.
// startPos - array of relative start positions of each digit per block
// out - output array. Everythng here is going to be sorted within the block at the end
// outPos - output positions
__global__ void sortRadixPerBlock(const unsigned int * const in, 
                                     const unsigned int * const inPos, 
                                     const size_t size, unsigned char shift, 
                                     unsigned char * const rel, 
                                     unsigned int * const startPos, 
                                     unsigned int * const out, 
                                     unsigned int * const outPos)
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
  
  // how many digits of each kind do we have?
  for(int i = 0; i < RADIX; i++)
  {
    // remember how many of the digits of this kind we have encountered
    if (i == digit_pos)
    {
      atomicAdd(&startPos[blockIdx.x + digit_pos], 1);
      break;
    }
  }
  __syncthreads();

  // initialize to the "startup" positions for each digit.
  if (digit_pos > 0)
  {
    relativePosition[threadIdx.x + digit_pos * BLOCK_SIZE] = threadIdx.x == 0 ? 0 : startPos[blockIdx.x + digit_pos - 1];
  }

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

  // now move things around
  out[rel[idx]] = in[idx];
  outPos[rel[idx]] = inPos[idx];
}

// Convert relative postions to absolute ones by adding up all the histograms we now have in the
// relPos array
// n blocks of RADIX threads each
// absPos, relPos - histograms of digit destribution
// relPos - relative to each of the sorting blocks (n)
// absPos - relative to the start of the array
__global__ void relPosToAbsPos(const unsigned int * const relPos, unsigned int * const absPos)
{
  if (blockIdx.x == 0)
  {
    return;
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int prevIdx = idx - blockDim.x; //(blockIdx.x - 1) * blockDim.x + threadIdx.x
  
  atomicAdd(&absPos[threadIdx.x], relPos[prevIdx]);
}

__global__ void moveToCorrectPos(unsigned int * const dest, 
                                 const unsigned int * const source, 
                                 size_t size,
                                 unsigned int shift,
                                 unsigned int * const destPos, 
                                 const unsigned int * const sourcePos, 
                                 const unsigned int * absPos,
                                 const unsigned int * relPos)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx > size)
  {
    return;
  }

  unsigned int mask = (NUM_BITS - 1) << shift;
  int digit = (source[idx] & mask) >> shift;

  // moves numbers sorted within a block to their rightful place
  // here threadIdx.x marks the right spot
  int destIndex = absPos[digit] + relPos[blockIdx.x * RADIX + digit] + threadIdx.x;

  dest[destIndex] = source[idx];
  destPos[destIndex] = sourcePos[idx];
  
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
}
