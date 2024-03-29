#include <stdio.h>
#include <cstdlib> 
#include <cstring> 
#include "timer.h"
#include <iostream>

#define blockSize 256
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int i = threadIdx.x + (blockIdx.x*blockDim.x);
    int stride = blockDim.x*gridDim.x;
    while(i<N*stride){
        if(i<N*stride){
        C[i] = A[i]+B[i];
        }
        i+=stride;
    }
}
float* d_A; 
float* d_B; 
float* d_C; 
float* h_A; 
float* h_B; 
float* h_C;
void Cleanup(bool);
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <scenario> <size in millions>\n", argv[0]);
        printf("Available scenarios:\n");
        printf("1 - One block with 1 thread\n");
        printf("2 - One block with 256 threads\n");
        printf("3 - Multiple blocks with 256 threads per block\n");

        return 1;
     }
     int scenario = atoi(argv[1]);
     if (scenario < 1 || scenario > 3) {
         printf("Invalid scenario number. Please choose a scenario between 1 and 3.\n");
         return 1;
     }
     int sizeInMillions = atoi(argv[2]);
     if (sizeInMillions <= 0) {
         printf("Invalid size. Size must be a positive integer.\n");
         return 1;
     }
     int million = 1000000;
     int N = sizeInMillions*million; //ValuesPerThread * GridWidth * BlockWidth;
     printf("Total vector size: %d\n", N); 
     // size_t is the total number of bytes for a vector.
     size_t size = N * sizeof(float);
 
     // Tell CUDA how big to make the grid and thread blocks.
     // Since this is a vector addition problem,
     // grid and thread block are both one-dimensional.
     //dim3 dimGrid/;/(1);                    
     //dim3 dimBlock;//(1);                 
     // Set up execution configuration
     dim3 dimBlock, dimGrid;
     if (scenario == 1) {
         dimGrid = dim3(1); // One block
         dimBlock = dim3(1); // One thread
     } else if (scenario == 2) {
         dimGrid = dim3(1); // One block
         dimBlock = dim3(blockSize); // 256 threads per block
     } else { // Scenario 3
         dimGrid = dim3((size + blockSize - 1) / blockSize); // Adjust number of blocks based on size
         dimBlock = dim3(blockSize); // 256 threads per block
     }
     
     // Allocate input vectors h_A and h_B in host memory
     h_A = (float*)malloc(size);
     if (h_A == 0) Cleanup(false);
     h_B = (float*)malloc(size);
     if (h_B == 0) Cleanup(false);
     h_C = (float*)malloc(size);
     if (h_C == 0) Cleanup(false);
 
     // Allocate vectors in device memory.
     cudaError_t error;
     error = cudaMalloc((void**)&d_A, size);
     if (error != cudaSuccess) Cleanup(false);
     error = cudaMalloc((void**)&d_B, size);
     if (error != cudaSuccess) Cleanup(false);
     error = cudaMalloc((void**)&d_C, size);
     if (error != cudaSuccess) Cleanup(false);
    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = 1;//(float)i;
     h_B[i] = 1;//(float)(N-i);   
    }
     // Warm up
     AddVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
     error = cudaGetLastError();
     if (error != cudaSuccess) Cleanup(false);
     cudaDeviceSynchronize();
 
    // Initialize timer
  
    initialize_timer();
    start_timer();

    AddVectors<<<dimGrid, dimBlock>>>(h_A, h_B, h_C, N);
    stop_timer();
    double time = elapsed_time();

    printf( "Time: %lf (sec)\n", 
             time);
    //printf( "Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
    //         time, nGFlopsPerSec, nGBytesPerSec);
    Cleanup(true);
}


void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}