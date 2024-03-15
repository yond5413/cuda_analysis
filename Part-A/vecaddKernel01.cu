/*
/// vecAddKernel00.cu
/// For COMS E6998 Spring 2024
/// Instructor: Kaoutar El Maghraoui
/// By Yonathan Daniel 
Created: 2024-03-14
/// This Kernel adds two Vectors A and B in C on GPU
/// with coalesced memory access.
/// 
*/
#include <stdio.h>
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    /*
    from lecture slides
    gridDim provides number of blocks
    blocks provides number of threads
    */
    //int foo = blockDim.x*gridDim.x*N; // Number of vector elements?
    int i = threadIdx.x + (blockIdx.x*blockDim.x);
    int stride = blockDim.x*gridDim.x;
    //printf("blockDim: %d, gridDim: %d, i:%d, N:%d, foo: %d, tid: %d, blockid: %d \n",blockDim.x,gridDim.x,i, N, foo,threadIdx.x,blockIdx.x);
    //printf("tid: %d, blockid: %d \n", threadIdx.x,blockIdx.x);
    //while(i<N){
    while(i<N*stride){
        if(i<N*stride){
        C[i] = A[i]+B[i];
        printf("i: %d, i+stride: %d, N: %d, foo:%d %d\n",i,(i+stride),N,blockDim.y,gridDim.y);
        }
        i+=stride;
    }
}
    