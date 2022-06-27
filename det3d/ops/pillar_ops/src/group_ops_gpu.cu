//
// Created by sgs on 2021/6/25.
//

#include "group_ops_gpu.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void flattenIndicePairsKernel(int N, int K, const int *indicePairs, const int *position,
                                         int *firstIndices, int *secondIndices) {
    // indicePairs: (N, K) -1[none]  position: (N*K, )  -1[none]
    // firstIndices: (L,)            secondIndices: (L,)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * K) return;

    indicePairs += tid;
    position += tid;
    if (indicePairs[0] < 0) return;

    int index = position[0];
    firstIndices[index] = tid / K;
    secondIndices[index] = indicePairs[0];
}

void flatten_indice_paris_kernel_launcher(int N, int K, const int *indicePairs, const int *position,
                                          int *firstIndices, int *secondIndices) {
    cudaError_t err;

    dim3 blocks(DIVUP(N*K, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    flattenIndicePairsKernel<<<blocks, threads>>>(N, K, indicePairs, position, firstIndices, secondIndices);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed (flattenIndicePairsKernel): %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_feature_kernel(int L, int C, const int *set_indices, const float *features, float *out){
    // set_indices: (L,)         features: (N, C)
    // out: (L, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    set_indices += tid;
    out += tid * C;
    features += set_indices[0] * C;
    for (int c = 0; c < C; ++c){
        out[c] = features[c];
    }
}

__global__ void gather_feature_grad_kernel(int L, int C, const int *set_indices, const float *outGrad, float *inGrad){
    // set_indices: (L,)         grad_features: (L, C)
    // grad_out: (N, C)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;

    set_indices += tid;
    outGrad += tid * C;
    inGrad += set_indices[0] * C;
    for (int c = 0; c < C; ++c){
        atomicAdd(inGrad + c, outGrad[c]);
    }
}

void gather_feature_kernel_launcher(int L, int C, const int *set_indices, const float *features, float *out){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_kernel<<<blocks, threads>>>(L, C, set_indices, features, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void gather_feature_grad_kernel_launcher(int L, int C, const int *set_indices, const float *outGrad, float *inGrad){
    cudaError_t err;
    dim3 blocks(DIVUP(L, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_feature_grad_kernel<<<blocks, threads>>>(L, C, set_indices, outGrad, inGrad);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
