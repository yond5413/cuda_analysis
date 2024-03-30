#include <stdio.h>
#include <cstdlib> 
#include <cstring> 
#include "timer.h"
#include <iostream>
///////////////////
#include <cudnn.h>
///////////////////

#define H 1024
#define W 1024
#define C 3 //input channels
#define FW 3
#define FH 3
#define K 64 //output channels
#define P 1 //padding 
double *d_I,*d_F,*d_O, *h_I,*h_F,*h_O;
double *d_Io,*h_Io;



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
    h_I = (double*)malloc(size_I*sizeof(double));
    h_F = (double*)malloc(size_F*sizeof(double));
    h_O = (double*)malloc(size_O*sizeof(double));
    h_Io = (double*)malloc(size_Io*sizeof(double));
   ///////////////////////////////////
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

        cudaMalloc(&d_I,size_I*sizeof(double));
        cudaMalloc(&d_F,size_F*sizeof(double));
        cudaMalloc(&d_O,size_O*sizeof(double));
        cudaMalloc(&d_Io,size_Io*sizeof(double));
    
        cudaMemcpy(d_Io,h_Io,size_Io * sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_F,h_F,size_F * sizeof(double),cudaMemcpyHostToDevice);
////////////////////////////////////////
    /*
    https://docs.nvidia.com/deeplearning/cudnn/api/cudnn-cnn-library.html
    */
    cudnnHandle_t c3;
    cudnnCreate(&c3);
    printf("Created c3?\n");
    cudnnTensorDescriptor_t in_descript, out_descript;
    cudnnFilterDescriptor_t filter_descript;
    cudnnConvolutionDescriptor_t convo_descript;

    cudnnCreateTensorDescriptor(&in_descript);
    cudnnCreateTensorDescriptor(&out_descript);
    cudnnCreateFilterDescriptor(&filter_descript);
    cudnnCreateConvolutionDescriptor(&convo_descript);
    
    cudnnConvolutionFwdAlgoPerf_t prefered_convo_alg;
    int alg;
    cudnnGetConvolutionForwardWorkspaceSize(c3,in_descript,filter_descript,convo_descript,,out_descript,size_O);
    //cudnnGetConvolutionForwardAlgorithm_v7(c3,in_descript,filter_descript,convo_descript,
    //out_descript,1,&alg,&prefered_convo_alg);
    cudnnConvolutionFwdAlgo_t algo = prefered_convo_alg.algo;
    double alpha = 1.0, beta = 0.0;
    initialize_timer();
    start_timer();
    
    cudnnConvolutionForward(c3,&alpha,in_descript,d_Io,filter_descript,d_F,
    convo_descript,algo, nullptr,0,&beta,out_descript,d_O);
    stop_timer();
    cudaMemcpy(h_O, d_O, sizeof(double) * size_O, cudaMemcpyDeviceToHost);
    
    double time = elapsed_time();
    time = time* 1000;
    double res = checksum(h_O);
    //printf( "checkSum: %lf\ntime: %lf\n", res,  time);
    printf("Checksum: %lf, ExcecutionTime: %lf (ms)\n",res,time);
    return 0;
}