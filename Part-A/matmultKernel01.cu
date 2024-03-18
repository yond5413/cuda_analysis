/*
For COMS E6998(HPML) Spring 2024
Modified by Yonathan Daniel
*/
#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE
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
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  // Each thread computes four elements of Csub
  // by accumulating results into Cvalue0, Cvalue1, Cvalue2, and Cvalue3
  float Cvalue0 = 0;
  float Cvalue1 = 0;
  float Cvalue2 = 0;
  float Cvalue3 = 0;
  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;
  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
      // Get sub-matrix Asub of A
      Matrix Asub = GetSubMatrix(A, blockRow, m);
      // Get sub-matrix Bsub of B
      Matrix Bsub = GetSubMatrix(B, m, blockCol);
      // Shared memory used to store Asub and Bsub respectively
      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix
      As[row][col] = GetElement(Asub, row, col);
      Bs[row][col] = GetElement(Bsub, row, col);
      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation
      __syncthreads();
      // Multiply Asub and Bsub together
      for (int e = 0; e < BLOCK_SIZE; ++e) {
          float tempA = As[row][e];
          float tempB0 = Bs[e][col];
          float tempB1 = Bs[e][col + 1];
          float tempB2 = Bs[e][col + 2];
          float tempB3 = Bs[e][col + 3];
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
  SetElement(Csub, row, col, Cvalue0);
  SetElement(Csub, row, col + 1, Cvalue1);
  SetElement(Csub, row + 1, col, Cvalue2);
  SetElement(Csub, row + 1, col + 1, Cvalue3);
}