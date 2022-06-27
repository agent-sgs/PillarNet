//
// Created by sgs on 2021/7/13.
//

#include "pillar_ops_gpu.h"

//#define assert_expr(x) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, ""); }
//#define assert_expr_msg(x, msg) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, msg); }
//void assert_fail(const char *str, const char *file, int line, std::string msg) {
//	std::cerr << "Assertion failed " << str << " " << file << ":" << line << " " << msg << std::endl;
//	abort();
//}

int create_pillar_indices_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                        at::Tensor pillar_mask_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(pillar_mask_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
	assert(C == 3);
    int B = pillar_mask_tensor.size(0);
    int H = pillar_mask_tensor.size(1);
    int W = pillar_mask_tensor.size(2);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    bool *pillar_mask = pillar_mask_tensor.data_ptr<bool>();

    create_pillar_indices_stack_kernel_launcher(N, B, H, W, bev_size, xyz, xyz_batch_cnt, pillar_mask);

    return 1;
}

int create_pillar_indices_wrapper(at::Tensor bev_indices_tensor, at::Tensor pillar_indices_tensor) {
	// pillar_indices_tensor: (M, 3) [byx]     bev_indices_tensor: (B, H, W)
	CHECK_INPUT(bev_indices_tensor);
	CHECK_INPUT(pillar_indices_tensor);

	int B = bev_indices_tensor.size(0);
	int H = bev_indices_tensor.size(1);
	int W = bev_indices_tensor.size(2);

	const int *bevIndices = bev_indices_tensor.data_ptr<int>();
	int *pillarIndices = pillar_indices_tensor.data_ptr<int>();

	create_pillar_indices_kernel_launcher(B, H, W, bevIndices, pillarIndices);

	return 1;
}

int create_pillar_indice_pairs_stack_wrapper(float bev_size, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
                                                at::Tensor pillar_bev_indices_tensor, at::Tensor indice_pairs_tensor) {
	CHECK_INPUT(xyz_tensor);
	CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(pillar_bev_indices_tensor);
    CHECK_INPUT(indice_pairs_tensor);

    int N = xyz_tensor.size(0);
    int C = xyz_tensor.size(1);
    assert(C == 3);
    int B = pillar_bev_indices_tensor.size(0);
    int H = pillar_bev_indices_tensor.size(1);
    int W = pillar_bev_indices_tensor.size(2);

    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *pillar_bev_indices = pillar_bev_indices_tensor.data_ptr<int>();
    int *indice_pairs = indice_pairs_tensor.data_ptr<int>();

    create_pillar_indice_pairs_stack_kernel_launcher(N, B, H, W, bev_size, xyz, xyz_batch_cnt,
													 pillar_bev_indices, indice_pairs);
    return 1;
}

