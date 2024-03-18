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
  /*// matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  //????
  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];
  */
  // Block row and column
  /*int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Each thread block computes one sub-matrix Csub of C
  //Matrix Csub = getSubMatrix(C, blockRow, blockCol);*/
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];
  // Each thread computes four elements of Csub
  // by accumulating results into Cvalue0, Cvalue1, Cvalue2, and Cvalue3
  float Cvalue0 = 0;
  float Cvalue1 = 0;
  float Cvalue2 = 0;
  float Cvalue3 = 0;
  // Thread row and column within Csub
  //int row = threadIdx.y;
  //int col = threadIdx.x;
  printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
      // Get sub-matrix Asub of A
      //Matrix Asub = getSubMatrix(A, blockRow, m);
      // Get sub-matrix Bsub of B
      //Matrix Bsub = getSubMatrix(B, m, blockCol);
      // Shared memory used to store Asub and Bsub respectively
      Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
      Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];
      //Bsub =&B.elements[B.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
      __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix
      shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];//getElement(Asub, row, col);
      shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation
      __syncthreads();
      // Multiply Asub and Bsub together
      #pragma unroll
      for (int e = 0; e < BLOCK_SIZE; ++e) {
          float tempA = shared_A[thread_row][e];
          float tempB0 = shared_B[e][thread_col];
          float tempB1 = shared_B[e][thread_col + 1];
          float tempB2 = shared_B[e][thread_col + 2];
          float tempB3 = shared_B[e][thread_col + 3];
          Cvalue0 += tempA * tempB0;
          Cvalue1 += tempA * tempB1;
          Cvalue2 += tempA * tempB2;
          Cvalue3 += tempA * tempB3;
      }
      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  Csub[thread_row * C.stride + thread_col] = Cvalue0;
    Csub[thread_row * C.stride + thread_col + BLOCK_SIZE] = Cvalue1;
Csub[thread_row * C.stride + thread_col + 2 * BLOCK_SIZE] = Cvalue2;
Csub[thread_row * C.stride + thread_col + 3 * BLOCK_SIZE] = Cvalue3;

  // second idea
 // Csub[thread_row * C.stride + thread_col] = Cvalue0;
  //Csub[thread_row * C.stride*2 + thread_col] = Cvalue1;
  //Csub[thread_row * C.stride*3 + thread_col] = Cvalue2;
  //Csub[thread_row * C.stride*4 + thread_col] = Cvalue3;
  // origninal below:
  /*setElement(Csub, row, col, Cvalue0);
  setElement(Csub, row, col + 1, Cvalue1);
  setElement(Csub, row + 1, col, Cvalue2);
  setElement(Csub, row + 1, col + 1, Cvalue3);*/
}