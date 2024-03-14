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

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
        printf("N = %d, &N = %p, i = %d, &i = %p\n", N, &N, i, &i);
}