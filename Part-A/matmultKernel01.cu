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
/*
within given block
how to losf into shared mem?

instead of having like a square/rectangular filter
load element from subarray and compute that way

A = | 1 | 2 |
    | 3 | 4 |

To compute C_sub
->
Main difference thread to compute offsets in block
    */
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
 // matrix blocks
 float *Asub, *Bsub, *Csub;
 // Putting these into registers speeds access.
 int thread_row = threadIdx.y;
 int thread_col = threadIdx.x;

//this is where the thread blocks start in the thread space
 int block_row = blockIdx.y;
 int block_col = blockIdx.x;

//this is where they go into memory
 int footprint_row = block_row*2;
 int footprint_col = block_col*2;

//4 different C values
 float Cval0 = 0;
 float Cval1 = 0;
 float Cval2 = 0;
 float Cval3 = 0;



 Csub = &C.elements[C.stride * footprint_row * FOOTPRINT_SIZE + FOOTPRINT_SIZE * footprint_col];

 for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){


 Asub = &A.elements[A.stride * FOOTPRINT_SIZE * footprint_row + FOOTPRINT_SIZE * m];
 Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * footprint_col];




 
 __shared__ float shared_A[FOOTPRINT_SIZE*FOOTPRINT_SIZE];
 __shared__ float shared_B[FOOTPRINT_SIZE*FOOTPRINT_SIZE];

 int offset = 0;

 int Arow = A.stride * thread_row/2;
 int Brow = B.stride * thread_row/2;
 int Acol = thread_col + thread_row%2 * BLOCK_SIZE;
 int Bcol = thread_col + thread_row%2 * BLOCK_SIZE;


 int sharedArow = thread_row/2 * FOOTPRINT_SIZE;
 int sharedBrow = thread_row/2 * FOOTPRINT_SIZE;
 int wFact = BLOCK_SIZE/2 * A.stride;
 int bFact = FOOTPRINT_SIZE * FOOTPRINT_SIZE;

 shared_A[sharedArow + Acol + offset*bFact] = Asub[Arow + Acol + offset*wFact];
 shared_B[sharedBrow + Bcol + offset*bFact] = Bsub[Brow + Bcol + offset*wFact];
 
 offset++;

 shared_A[sharedArow + Acol + offset*bFact] = Asub[Arow + Acol + offset*wFact];
 shared_B[sharedBrow + Bcol + offset*bFact] = Bsub[Brow + Bcol + offset*wFact];
 
 
 offset++;

 shared_A[sharedArow + Acol + offset*bFact] = Asub[Arow + Acol + offset*wFact];
 shared_B[sharedBrow + Bcol + offset*bFact] = Bsub[Brow + Bcol + offset*wFact];  
 
 offset++;

 shared_A[sharedArow + Acol + offset*bFact] = Asub[Arow + Acol + offset*wFact];
 shared_B[sharedBrow + Bcol + offset*bFact] = Bsub[Brow + Bcol + offset*wFact];
 
// sync threads before updating Cvals
  __syncthreads();



#pragma unroll
   for( int e = 0; e<FOOTPRINT_SIZE; e++){
//top left and top right
 Cval0 += shared_A[e*FOOTPRINT_SIZE + thread_col] * shared_B[thread_row + e ];
 Cval1 += shared_A[e*FOOTPRINT_SIZE + thread_col + BLOCK_SIZE] * shared_B[thread_row + e + BLOCK_SIZE];
//bot left and bot right
 Cval2 += shared_A[e*FOOTPRINT_SIZE + thread_col + BLOCK_SIZE*FOOTPRINT_SIZE] * shared_B[thread_row + e + BLOCK_SIZE*FOOTPRINT_SIZE];
 Cval3 += shared_A[e*FOOTPRINT_SIZE + thread_col + BLOCK_SIZE*FOOTPRINT_SIZE + BLOCK_SIZE] * shared_B[thread_row + e + BLOCK_SIZE*FOOTPRINT_SIZE + BLOCK_SIZE];
   }

   // Synchronize to ensure all Cvalues have been incremented
   // before reading in the next shared_A AND shared_B BLOCKS
   __syncthreads();




 }
 // Write Csub to GLOBAL memory.
 // Each thread writes its own cell value.
 Csub[thread_row * C.stride + thread_col  ] = Cval0;
 Csub[thread_row * C.stride + thread_col  + BLOCK_SIZE] = Cval1;
 Csub[thread_row * C.stride + thread_col  + BLOCK_SIZE*C.stride] = Cval2;
 Csub[thread_row * C.stride + thread_col  + BLOCK_SIZE*C.stride + BLOCK_SIZE] = Cval3;
}