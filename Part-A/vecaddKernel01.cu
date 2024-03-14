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
    //int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    int i = threadIdx.x;
    if(i<N)
    C[i] = A[i]+B[i]
    
}