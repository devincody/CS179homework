/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    while (tid < padded_length){
        cufftComplex in = raw_data[tid];
        cufftComplex impulse = impulse_v[tid];

        out_data[tid].x = (in.x*impulse.x - in.y*impulse.y)/padded_length;
        out_data[tid].y = (in.x*impulse.y + in.y*impulse.x)/padded_length;

        tid += blockDim.x*gridDim.x;
    }

}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    // Thread index for accessing global memory
    int indx = blockIdx.x*blockDim.x + threadIdx.x;
    // Thread index for accessing shared memory made const for speedup
    const int tid = threadIdx.x;

    // Shared memory allocated dynamically from kernal call
    extern __shared__ float sdata[];

    // Pull (real) data into shared memory
    // Memory is coalleced
    sdata[tid] = abs(out_data[indx].x);
    indx += gridDim.x*blockDim.x;

    while (indx < padded_length){ // if there are more data values than total threads,
        sdata[tid] = max(abs(out_data[indx].x), sdata[tid]); // use max to avoid if statement
        indx += gridDim.x*blockDim.x;
    }

    __syncthreads(); // make sure shared memory is ready

    for (int s= blockDim.x/2; s > 0; s>>=1){
        /*
        This implementation uses the tecnique of sequential addressing.
        Each thread is responsible for finding the max between the data
        at tid and tid+s. This approach allows us to avoid bank conflicts
        since the stride is always 1 
        */
        if(tid < s){
            // s = 16, 8, 4, 2, 1
            // here each thread finds the max between the data at tid and
            // and address on the "other side", a distance of s away.
            sdata[tid] = max(sdata[tid+s],sdata[tid]);
        }
        __syncthreads();
    }

    // In emperical tests, unrolling the loop did NOT result in speed gains
    // if (tid < 16){
    //     sdata[tid] = max(sdata[tid+16],sdata[tid]);
    //     sdata[tid] = max(sdata[tid+8],sdata[tid]);
    //     sdata[tid] = max(sdata[tid+4],sdata[tid]);
    //     sdata[tid] = max(sdata[tid+2],sdata[tid]);
    //     sdata[tid] = max(sdata[tid+1],sdata[tid]);
    // }
    // __syncthreads();

    // atomicMax is used by each thread to compare the value of the data point
    // at the first index (i.e. the max for the particular warp) with the current
    // maximum value in global memory.
    if (tid == 0) atomicMax(max_abs_val, sdata[0]);

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    int tid = blockDim.x*blockIdx.x +  threadIdx.x;

    while (tid < padded_length){
        out_data[tid].x /= *max_abs_val;
        out_data[tid].y /= *max_abs_val;

        tid += gridDim.x*blockDim.x;
    } 

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);

}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length);

    // Dynamic shared memory allocation?
    // Do we need anothe array for 

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
