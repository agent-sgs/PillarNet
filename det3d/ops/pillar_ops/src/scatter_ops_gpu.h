//
// Created by sgs on 2021/6/21.
//

#ifndef SCATTER_OPS_GPU_H
#define SCATTER_OPS_GPU_H

#include "cuda_utils.h"


int scatter_max_wrapper(at::Tensor index_tensor, at::Tensor src_tensor, at::Tensor arg_tensor, at::Tensor out_tensor);

int scatter_max_grad_wrapper(at::Tensor arg_tensor, at::Tensor grad_out_tensor, at::Tensor grad_src_tensor);

void scatter_max_kernel_launcher(int C, int L, int M, const int *index, const float *src, int *arg, float *out);

void scatter_max_grad_kernel_launcher(int C, int M, const int *arg, const float *grad_out, float *grad_src);

#endif //SCATTER_OPS_GPU_H
