//
// Created by sgs on 2021/7/13.
//

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define assert_expr(x) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, ""); }
#define assert_expr_msg(x, msg) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, msg); }
inline void assert_fail(const char *str, const char *file, int line, std::string msg) {
    std::cerr << "Assertion failed " << str << " " << file << ":" << line << " " << msg << std::endl;
    abort();
}

#endif //CUDA_UTILS_H
