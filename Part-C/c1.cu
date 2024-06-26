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
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index
    int out_c = blockIdx.z; // Output channel index
    double O_value = 0.0;
    // Perform convolution for each pixel
    for (int i = 0; i < FH; ++i) {
        for (int j = 0; j < FW; ++j) {
            for (int c = 0; c < C; ++c) {
                // computing hte offsets here 
                int row_off = row- P + i;   
                int col_off = col - P + j;
                // Check if the pixel is within the input image boundaries
                if (row_off >= 0 && row_off < H && col_off >= 0 && col_off < W) {
                    O_value += I[c * (H * W) + row_off * W + col_off] * F[out_c* (C * FH * FW) + c * (FH * FW) + i * FW + j];
                }
            }
        }
    }
    // Store the result in the output tensor
    O[out_c* (H * W) + row * W + col] = O_value;
}

double checksum(double* O) {
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }
    return checksum;
}

int main(int argc, char* argv[]){
    
    size_t size_I = H*W*C;
    size_t size_Io =  (H+2*P)*(W+2*P)*C;
    size_t size_F = FH*FW*C*K;
    size_t size_O = K*H*W;
    //printf("Malloc \n");
    h_I = (double*)malloc(size_I*sizeof(double));
    h_F = (double*)malloc(size_F*sizeof(double));
    h_O = (double*)malloc(size_O*sizeof(double));
    h_Io = (double*)malloc(size_Io*sizeof(double));
    //printf("init?\n");
    // init I tensor
    // Initialize I
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                h_I[c * W * H + x * H + y] = c * (x + y);
            }
        }
    }
    //printf("init->F\n");
    // Initialize F filter
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    h_F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }
    //printf("init->Io\n");
    // Initialize I0 with padding
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W + 2 * P; ++x) {
            for (int y = 0; y < H + 2 * P; ++y) {
                if (x == 0 || y == 0 || x == W + 2 * P - 1 || y == H + 2 * P - 1) {
                    h_Io[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = 0;
                } else {
                    h_Io[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = h_I[c * W * H + (x - 1) * H + (y - 1)];
                }
            }
        }
    }
   //printf("cuddda \n");
    cudaMalloc(&d_I,size_I*sizeof(double));
    cudaMalloc(&d_F,size_F*sizeof(double));
    cudaMalloc(&d_O,size_O*sizeof(double));
    cudaMalloc(&d_Io,size_Io*sizeof(double));

    cudaMemcpy(d_Io,h_Io,size_Io * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,h_F,size_F * sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimBlock(H); //1024
    dim3 dimGrid(K,H); //64,1024
    // warm-up
    //printf("HIIII \n");
    convolution<<<dimGrid, dimBlock>>>(d_Io, d_F, d_O);
    cudaDeviceSynchronize();
    //
    //printf("warmupppp done");
    initialize_timer();
    start_timer();
    convolution<<<dimGrid, dimBlock>>>(d_Io, d_F, d_O);
    cudaDeviceSynchronize();
    stop_timer();
    double time = elapsed_time();
    time = time* 1000;
    //printf( "Time: %lf (sec)\n",time);
    cudaMemcpy(h_O, d_O, size_O*sizeof(double), cudaMemcpyDeviceToHost);
    double res = checksum(h_O);
    //printf("res: %lf \n", res);
    //printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n",
    //time, nFlops, nGFlopsPerSec);
    printf("Checksum: %lf, ExcecutionTime: %lf  (ms) \n",res,time);
    cudaFree(d_I);
    cudaFree(d_Io);
    cudaFree(d_F);
    cudaFree(d_O);
    return 0;
}