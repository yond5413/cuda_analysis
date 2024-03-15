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
    int i = threadIdx.x + (blockIdx.x*blockDim.x);
    printf("blockDim: %d, gridDim:%d",blockDim.x,gridDim.x);
    while(i<N){
        C[i] = A[i]+B[i];
        i+= blockDim.x*gridDim.x;
    }
}
    