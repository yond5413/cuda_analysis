/*
For COMS E6998(HPML) Spring 2024
Modified by Yonathan Daniel
*/
#include <stdio.h> // for debugging

#include "matmultKernel.h"

//#define FOOTPRINT_SIZE BLOCK_SIZE
// Define a gpu kernel to perform matrix multiplication
// of A x B = C.

////// Helper functions start//////  
__device__ float getElement(const Matrix A, int row, int col){
return A.elements[row*A.stride+col];
}
//
__device__ void setElement(Matrix A, int row, int col, float val){
A.elements[row*A.stride*BLOCK_SIZE +col*BLOCK_SIZE] = val;
}

__device__ Matrix getSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
///// Helper end functions end/////

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
   // Matrix blocks
   float *Asub, *Bsub, *Csub;

   // Putting these into registers speeds access.
   int thread_row = threadIdx.y;
   int thread_col = threadIdx.x;
   int block_row = blockIdx.y;
   int block_col = blockIdx.x;

   // Each THREAD BLOCK computes one submatrix Csub of C
   // EACH THREAD creates its own matrix descriptor Csub
   Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

   // Each thread computes one element of Csub in its copy of Cvalue
   float Cvalue[4] = {0.0f, 0.0f, 0.0f, 0.0f};

   // Loop over all submatrices in block_row of A and block_col of B
   // required to compute Csub. Block multiply each pair of submatrices
   // and accumulate results
   for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
       // Get Asub and Bsub descriptors
       Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
       Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

       // Copy ELEMENTS OF Asub and Bsub into shared memory
       // EACH THREAD loads ONE ELEMENT of Asub and ONE of Bsub
       // Notice: it does not need to be the element it requires to
       //         compute its Cvalue, as long as all elements are
       //         collaboratively read.

       // Notice: every thread declares shared_A and shared_B in shared memory
       //         even though a thread block has only one shared_A and one shared_B
       __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
       __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

       // Each thread copies just one element of shared_A and shared_B
       shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
       shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

       // Synchronize to ensure all elements are read
       __syncthreads();

       // Do an inproduct of one row of shared_A and one col of shared_B
       // computing one Cvalue by accumulation
#pragma unroll
       for (int e = 0; e < BLOCK_SIZE; ++e) {
           Cvalue[0] += shared_A[thread_row][e] * shared_B[e][thread_col];
           Cvalue[1] += shared_A[thread_row + 8][e] * shared_B[e][thread_col];
           Cvalue[2] += shared_A[thread_row + 16][e] * shared_B[e][thread_col];
           Cvalue[3] += shared_A[thread_row + 24][e] * shared_B[e][thread_col];
       }

       // Synchronize to ensure all Cvalues have been incremented
       // before reading in the next shared_A AND shared_B BLOCKS
       __syncthreads();
   }

   // Write Csub to GLOBAL memory.
   // Each thread writes its own cell value.
   Csub[thread_row * C.stride + thread_col] = Cvalue[0];
   Csub[(thread_row + 8) * C.stride + thread_col] = Cvalue[1];
   Csub[(thread_row + 16) * C.stride + thread_col] = Cvalue[2];
   Csub[(thread_row + 24) * C.stride + thread_col] = Cvalue[3];
}