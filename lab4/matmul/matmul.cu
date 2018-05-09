#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
                                           
//helper_cuda.h contains the error checking macros. note that they're called
//CUDA_CALL and CUBLAS_CALL instead of the previous names


//TODO: perform the following matrix multiplications using cublas

#define M 2 // 
#define N 3
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main(int argc, char *argv[]) {
    double A[N * M] = {1, 2, 3, 4, 5, 6};
    double B[M * N] = {1, 2, 3, 4, 5, 6};
    double res1[N * N];
    double res2[M * M];

    //Create output arrays for gpu matricies
    double gpu_results1[N * N];
    double gpu_results2[M * M];

    int i, j, k;

    //TODO: cudaMalloc buffers, copy these to device, etc.

    //Generate handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Declare pointers
    double *d_A, *d_B, *d_res1, *d_res2;
    double alpha = 1, beta = 0;

    // Allocate Space on GPU
    cudaMalloc(&d_A, N*M*sizeof(double));
    cudaMalloc(&d_B, M*N*sizeof(double));
    cudaMalloc(&d_res1, N*N*sizeof(double));
    cudaMalloc(&d_res2, M*M*sizeof(double));

    // Transfer memory to GPU
    cudaMemcpy(d_A, A, N*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*M*sizeof(double), cudaMemcpyHostToDevice);

    // prep output arrays (not entirely necessary)
    cudaMemset(d_res1, 0, N*N*sizeof(double));
    cudaMemset(d_res2, 0, M*M*sizeof(double));

    // A * B
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf it to check


    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // Give handle, no operation on matricies
                N, N, M,                          // rows in output matrix, cols of output matrix, common dimension of input matricies
                &alpha,                           // Scale matrix multiplication product by one
                d_A, N,                           // Pointer to first matrix, leading dimension of first matrix
                d_B, M,                           // Pointer to second matrix, leading dimension of second matrix
                &beta,                            // Do not accumulate
                d_res1, N);                       // where to write, leading dim of output matrix

    cudaMemcpy(&gpu_results1, d_res1, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    printf("A * B\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res1[IDX2C(i, j, N)] = 0;
            for (k = 0; k < M; k++) {
                res1[IDX2C(i, j, N)] += A[IDX2C(i, k, N)] * B[IDX2C(k, j, M)];
            }
            printf("[%d, %d] = %f (cpu) and %f (gpu)\n", i, j, res1[IDX2C(i, j, N)], gpu_results1[IDX2C(i, j, N)]);
        }
    }



    // A^T * B^T
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf to check it

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,// Give handle, no operation on matricies
                M, M, N,// rows in output matrix, cols of output matrix, common dimension of input matricies
                &alpha,// Scale matrix multiplication product by one
                d_A, N,// Pointer to first matrix, leading dimension of first matrix
                d_B, M,// Pointer to second matrix, leading dimension of second matrix
                &beta,// Do not accumulate
                d_res2, M);// where to write, leading dim of output matrix

    for (int i = 0; i < M * N; ++i)
    {
        printf("A[%d]: %f\n", i, A[i]);
        printf("B[%d]: %f\n", i, B[i]);
    }

    cudaMemcpy(&gpu_results2, d_res2, M*M*sizeof(double), cudaMemcpyDeviceToHost);

    printf("\nA^T * B^T\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            res2[IDX2C(i, j, M)] = 0;
            for (k = 0; k < N; k++) {
                res2[IDX2C(i, j, M)] += A[IDX2C(k, i, N)] * B[IDX2C(j, k, M)];
            }
            printf("[%d, %d] = %f (cpu) and %f (gpu)\n", i, j, res2[IDX2C(i, j, M)], gpu_results2[IDX2C(i, j, M)]);
        }
    }



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res1);
    cudaFree(d_res2);

    cublasDestroy(handle);
}