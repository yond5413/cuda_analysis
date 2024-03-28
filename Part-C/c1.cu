#include <stdio.h>
#include <cstdlib> 
#include <cstring> 
#include "timer.h"
#include <iostream>


#define H 1024
#define W 1024
#define C 3 //input channels
#define FW 3
#define FH 3
#define K 64 //output channels
#define P 1 //padding 
double *d_I,*d_F,*d_O, *h_I,*h_F,*h_O;
double *d_Io,*h_Io;

__global__ void convolution(double *I,double *F, double *O){
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // Global column index
    int c_out_idx = blockIdx.z; // Output channel index

    double O_value = 0.0;

    // Perform convolution for each pixel
    for (int i = 0; i < FH; ++i) {
        for (int j = 0; j < FW; ++j) {
            for (int c = 0; c < C; ++c) {
                int row_offset = row_idx - P + i;
                int col_offset = col_idx - P + j;

                // Check if the pixel is within the input image boundaries
                if (row_offset >= 0 && row_offset < H && col_offset >= 0 && col_offset < W) {
                    O_value += I[c * (H * W) + row_offset * W + col_offset] * F[c_out_idx * (C * FH * FW) + c * (FH * FW) + i * FW + j];
                }
            }
        }
    }
    // Store the result in the output tensor
    O[c_out_idx * (H * W) + row_idx * W + col_idx] = O_value;
}


void initIo(double *I, double *Io,int padding){
    int paddedH = H + P*padding; //2 * padding;
    int paddedW = W + P* padding;
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < paddedH; ++i) {
            for (int j = 0; j < paddedW; ++j) {
                if (i < padding || i >= paddedH - padding || j < padding || j >= paddedW - padding) {
                    // Apply zero padding to the border regions
                    Io[c * paddedH * paddedW + i * paddedW + j] = 0.0;
                } else {
                    // Copy values from the original tensor I to the padded tensor Io
                    Io[c * paddedH * paddedW + i * paddedW + j] = I[c * H * W + (i - padding) * W + (j - padding)];
                }
            }
        }
    }
}

int main(int argc, char* argv[]){
    
    size_t size_I = H*W*C;
    size_t size_Io =  (H+2*P)*(W+2*P)*C;
    size_t size_F = FH*FW*C*K;
    size_t size_O = K*H*W;

    h_I = (double*)malloc(size_I);
    h_F = (double*)malloc(size_F);
    h_O = (double*)malloc(size_O);
    h_Io = (double*)malloc(size_Io);
    // init I tensor
    for(int c = 0;c<C;c++){
        for(int i = 0;i<H;++i){
            for(int j = 0;j<W;++j){
                h_I[c*H*W +i*W+j] = c*(i+j);
            }
        }
    }
    // init F filter
    for(int k = 0; k<K;++k){
        for(int c = 0;c<C;++c){
            for(int i = 0; i<FW;++i){
                for(int j = 0; j<FH;++j){
                    h_F[k*C*FH*FW +c*FW*FH+i*FH+j] = (c+k)*(i+j);
                    // iterator*nested branch cond 
                }
            }
        }
    }
   
    cudaMalloc(&d_I,size_I*sizeof(double));
    cudaMalloc(&d_F,size_F*sizeof(double));
    cudaMalloc(&d_O,size_O*sizeof(double));
    cudaMalloc(&d_Io,size_Io*sizeof(double));

    cudaMemcpy(d_Io,h_Io,size_Io * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,h_F,size_F * sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimBlock(H); //1024
    dim3 dimGrid(K,H); //64,1024
    // warm-up
    printf("HIIII \n");
    convolution<<<dimGrid, dimBlock>>>(d_Io, d_F, d_O);
    cudaDeviceSynchronize();
    //
    printf("warmupppp done");
    initialize_timer();
    start_timer();
    convolution<<<dimGrid, dimBlock>>>(d_Io, d_F, d_O);
    cudaDeviceSynchronize();
    stop_timer();
    double time = elapsed_time();
    printf( "Time: %lf (sec)\n",time);

    //printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n",
    //time, nFlops, nGFlopsPerSec);

    return 0;
}