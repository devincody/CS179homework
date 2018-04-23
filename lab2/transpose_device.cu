#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).


       vROW    v COL      vROW    vCOL
output[j + n * i] = input[i + n * j];

 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses
    /*
    This implementation is suboptimal in that it doesnt use shared memory.
    It instead relies on slow global memory transactions. Reading the 
    memory from global memory (i.e. from input[]) is coallesced, but the
    writing to global memory (i.e. to output[]) is not coallesced.
    */

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)      
        output[j + n * i] = input[i + n * j]; //global memory accesses
                                              //when moved to shared, there will be bank conflicts
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    /*
    Non-optimalities include the for loops not being unrolled, not using 
    const variables, and not putting most of the reads in one location.
    */

    __shared__ float data[64*65]; //Using shared memory to take advantage of coalleced memory transactions
                                  //

    int i =     threadIdx.x + 64 * blockIdx.x; //i = internal ROW of INPUT
    int j = 4 * threadIdx.y + 64 * blockIdx.y; // COL
    int end_j = j + 4;

    int ii =   threadIdx.x; //shared i
    int jj = 4*threadIdx.y; //shared j

    // Read the data
    for (; j < end_j; j++){ // Unroll this next time
        data[ii + 65*jj] = input[i + n * j]; 
        jj++;
    }

    // make sure all the threads have read the data before starting to write
    __syncthreads();


    i =     threadIdx.x + 64 * blockIdx.y; //global indicies
    j = 4 * threadIdx.y + 64 * blockIdx.x; //notice the blockIdx s have been swapped
    end_j = j + 4;

    ii = 4*threadIdx.y;
    jj =   threadIdx.x;

    // Write the data
    for(; j < end_j; j++){
        output[i + n * j] = data[ii + 65*jj];
        ii++;
    }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.

    // Initialize shared memory
    __shared__ float data[64*65];


    /* 
    For the input write code 
    */
    const int i =     threadIdx.x + 64 * blockIdx.x; // row in global memory
    const int j = 4 * threadIdx.y + 64 * blockIdx.y; // col in global memory

    const int ii =   threadIdx.x; // row in shared memory
    const int jj = 4*threadIdx.y; // col in shared memory


    /* 
    For the output write code 
    ii_ and jj_ are included for completeness (i.e. negigible speedup if removed)    
    */
    const int i_ =     threadIdx.x + 64 * blockIdx.y; // row in global memory
    const int j_ = 4 * threadIdx.y + 64 * blockIdx.x; // col in global memory

    const int ii_ = 4*threadIdx.y; // row in shared memory 
    const int jj_ =   threadIdx.x; // col in shared memory 

    // Store to shared memory
    data[ii + 65* jj     ] = input[i + n *  j    ]; 
    data[ii + 65*(jj + 1)] = input[i + n * (j + 1)]; 
    data[ii + 65*(jj + 2)] = input[i + n * (j + 2)]; 
    data[ii + 65*(jj + 3)] = input[i + n * (j + 3)]; 

    __syncthreads();

    // Write to global memory
    output[i_ + n *  j_     ] = data[ ii_      + 65*jj_];
    output[i_ + n * (j_ + 1)] = data[(ii_ + 1) + 65*jj_];
    output[i_ + n * (j_ + 2)] = data[(ii_ + 2) + 65*jj_];
    output[i_ + n * (j_ + 3)] = data[(ii_ + 3) + 65*jj_];

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
