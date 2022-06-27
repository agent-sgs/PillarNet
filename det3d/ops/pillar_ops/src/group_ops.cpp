//
// Created by sgs on 2021/6/25.
//
#include "group_ops_gpu.h"


int flatten_indice_pairs_wrapper(at::Tensor indice_pairs_tensor, at::Tensor position_tensor,
                                 at::Tensor first_indices_tensor, at::Tensor second_indices_tensor){
    // indice_pairs_tensor: (N, K)  position_tensor: (N*K) first_indices_tensor: (L,) second_indices_tensor: (L,)

    CHECK_INPUT(indice_pairs_tensor);
    CHECK_INPUT(position_tensor);
    CHECK_INPUT(first_indices_tensor);
    CHECK_INPUT(second_indices_tensor);

    int N = indice_pairs_tensor.size(0);
    int K = indice_pairs_tensor.size(1);

    const int *indicePairs = indice_pairs_tensor.data_ptr<int>();
    const int *position = position_tensor.data_ptr<int>();
    int *firstIndices = first_indices_tensor.data_ptr<int>();
    int *secondIndices = second_indices_tensor.data_ptr<int>();

    flatten_indice_paris_kernel_launcher(N, K, indicePairs, position, firstIndices, secondIndices);

    return 1;
}


int gather_feature_wrapper(at::Tensor set_indices_tensor, at::Tensor features_tensor, at::Tensor new_features_tensor){

    CHECK_INPUT(set_indices_tensor);
    CHECK_INPUT(features_tensor);
    CHECK_INPUT(new_features_tensor);

    int L = set_indices_tensor.size(0);
    int C = features_tensor.size(1);

    const int *set_indices = set_indices_tensor.data_ptr<int>();
    const float *features = features_tensor.data_ptr<float>();
    float *new_features = new_features_tensor.data_ptr<float>();

    gather_feature_kernel_launcher(L, C, set_indices, features, new_features);

    return 1;
}

int gather_feature_grad_wrapper(at::Tensor set_indices_tensor, at::Tensor grad_out_tensor, at::Tensor grad_features_tensor){

    CHECK_INPUT(set_indices_tensor);
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(grad_features_tensor);

    int L = set_indices_tensor.size(0);
    int C = grad_features_tensor.size(1);

    const int *set_indices = set_indices_tensor.data_ptr<int>();
    const float *outGrad = grad_out_tensor.data_ptr<float>();
    float *inGrad = grad_features_tensor.data_ptr<float>();

    gather_feature_grad_kernel_launcher(L, C, set_indices, outGrad, inGrad);

    return 1;
}