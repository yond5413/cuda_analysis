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
    int foo = blockDim.x*gridDim.x*N; // Number of vector elements?
    int i = threadIdx.x + (blockIdx.x*blockDim.x)*N;
    printf("blockDim: %d, gridDim: %d, i:%d, N:%d, foo: %d, tid: %d, blockid: %d \n",blockDim.x,gridDim.x,i, N, foo,threadIdx.x,blockIdx.x);
    //printf("tid: %d, blockid: %d \n", threadIdx.x,blockIdx.x);
    //while(i<N){
    //if(i<foo)//(N))
    C[i] = A[i]+B[i];
        //i+= blockDim.x*gridDim.x;
//}
}
    