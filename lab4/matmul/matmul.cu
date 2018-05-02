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
    float A[N * M] = {1, 2, 3, 4, 5, 6};
    float B[M * N] = {1, 2, 3, 4, 5, 6};
    float res1[N * N];
    float res2[M * M];

    int i, j, k;

    //TODO: cudaMalloc buffers, copy these to device, etc.

    cublasHandle_t handle;
    cublasCreate(&handle);

    double *d_A, *d_B, *d_res1, *d_res2;
    double alpha = 1, beta = 0;

    cudaMalloc(&d_A, N*M*sizeof(double));
    cudaMalloc(&d_B, M*N*sizeof(double));
    cudaMalloc(&d_res1, N*N*sizeof(double));
    cudaMalloc(&d_res2, M*M*sizeof(double));


    cudaMemcpy(d_A, A, N*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*M*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(d_res1, 0, N*N*sizeof(double));
    cudaMemset(d_res2, 0, M*M*sizeof(double));

    // A * B
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf it to check
    printf("A * B\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res1[IDX2C(i, j, N)] = 0;
            for (k = 0; k < M; k++) {
                res1[IDX2C(i, j, N)] += A[IDX2C(i, k, N)] * B[IDX2C(k, j, M)];
            }
            printf("[%f, %f] = %d\n", i, j, res1[IDX2C(i, j, N)]);
        }
    }

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, M, 
                &alpha, 
                d_A, N,
                d_B, M,
                &beta,
                d_res1, N);

    cudaMemcpy(&res1, d_res1, N*N*sizeof(double), cudaMemcpyDeviceToHost);

    // A^T * B^T
    // TODO: do this on GPU too with cuBLAS, copy result back, and printf to check it
    printf("\nA^T * B^T\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            res2[IDX2C(i, j, M)] = 0;
            for (k = 0; k < N; k++) {
                res2[IDX2C(i, j, M)] += A[IDX2C(k, i, N)] * B[IDX2C(j, k, M)];
            }
            printf("[%f, %f] = %d\n", i, j, res2[IDX2C(i, j, M)]);
        }
    }

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                M, M, N,
                &alpha,
                d_A, N,
                d_B, M,
                &beta,
                d_res2, M);

    cudaMemcpy(&res2, d_res2, M*M*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res1);
    cudaFree(d_res2);

    cublasDestroy(handle);
}