//
// Created by sgs on 2021/6/21.
//

#include "atomics.cuh"
#include "scatter_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void scatter_max_kernel(int C, int L, int M, const int *index, const float *src, float *out){
    // index: (L) src: (C, L)  out: (C, M)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= C * L) return;

    int cId = tid / L;
    int pId = tid % L;

    src += tid;
    index += pId;

    atomMax(out + cId * M + index[0], src[0]);
}

__global__ void scatter_arg_max_kernel(int C, int L, int M, const int *index, const float *src, const float *out, int *arg){
    // index: (L) src: (C, L)  out: (C, M)   arg: (C, M)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= C * L) return;

    int cId = tid / L;
    int pId = tid % L;

    src += tid;
    index += pId;

    int oId = cId * M + index[0];
    out += oId;
    arg += oId;

    if (abs(src[0] - out[0]) < 1e-5){
        arg[0] = tid;
    }
}

__global__ void scatter_max_grad_kernel(int C, int M, const int *arg, const float *grad_out, float *grad_src){
    // arg: (C, M)  grad_src: (C, L)  grad_out: (C, M)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= C * M) return;

    grad_out += tid;
    arg += tid;

    if (arg[0] < 0) return;
    grad_src[arg[0]] = grad_out[0];
}


void scatter_max_kernel_launcher(int C, int L, int M, const int *index, const float *src, int *arg, float *out) {
    cudaError_t err;
    dim3 blocks(DIVUP(C * L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    scatter_max_kernel<<<blocks, threads>>>(C, L, M, index, src, out);
    scatter_arg_max_kernel<<<blocks, threads>>>(C, L, M, index, src, out, arg);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (scatter_max_kernel or scatter_arg_max_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void scatter_max_grad_kernel_launcher(int C, int M, const int *arg, const float *grad_out, float *grad_src) {
    cudaError_t err;
    dim3 blocks(DIVUP(C * M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    scatter_max_grad_kernel<<<blocks, threads>>>(C, M, arg, grad_out, grad_src);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (scatter_max_grad_kernel) : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
