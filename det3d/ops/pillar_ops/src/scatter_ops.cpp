//
// Created by sgs on 2021/6/21.
//
#include "scatter_ops_gpu.h"


int scatter_max_wrapper(at::Tensor index_tensor, at::Tensor src_tensor, at::Tensor arg_tensor, at::Tensor out_tensor){
    // src: (C, L)   index: (L)  arg: (C, M)  out: (C, M)
    CHECK_INPUT(src_tensor);
    CHECK_INPUT(index_tensor);
    CHECK_INPUT(arg_tensor);
    CHECK_INPUT(out_tensor);

    int C = src_tensor.size(0);
    int L = src_tensor.size(1);
    int M = out_tensor.size(1);

    const float *src = src_tensor.data_ptr<float>();
    const int *index = index_tensor.data_ptr<int>();
    int * arg = arg_tensor.data_ptr<int>();
    float *out = out_tensor.data_ptr<float>();

    scatter_max_kernel_launcher(C, L, M, index, src, arg, out);
    return 1;
}

int scatter_max_grad_wrapper(at::Tensor arg_tensor, at::Tensor grad_out_tensor, at::Tensor grad_src_tensor){
    // arg: (C, M)   grad_out: (C, M)  grad_src: (C, L)
    CHECK_INPUT(arg_tensor);
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(grad_src_tensor);

    int C = grad_out_tensor.size(0);
    int M = grad_out_tensor.size(1);

    const int *arg = arg_tensor.data_ptr<int>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    float *grad_src = grad_src_tensor.data_ptr<float>();

    scatter_max_grad_kernel_launcher(C, M, arg, grad_out, grad_src);
    return 1;
}
