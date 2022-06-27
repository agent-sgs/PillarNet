//
// Created by sgs on 2021/7/13.
//

#ifndef PILLAR_QUERY_OPS_GPU_H
#define PILLAR_QUERY_OPS_GPU_H

#include "cuda_utils.h"


int create_pillar_indices_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
										at::Tensor pillar_mask_tensor);

int create_pillar_indices_wrapper(at::Tensor bev_indices_tensor, at::Tensor pillar_indices_tensor);

int create_pillar_indice_pairs_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
											 at::Tensor pillar_bev_indices_tensor, at::Tensor indice_pairs_tensor);

void create_pillar_indices_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
												 const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask);

void create_pillar_indices_kernel_launcher(int B, int H, int W, const int *bevIndices, int *pillarIndices);

void create_pillar_indice_pairs_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
													  const float *xyz, const int *xyz_batch_cnt,
													  const int *pillar_bev_indices, int *indice_pairs);

#endif //PILLAR_QUERY_OPS_GPU_H
