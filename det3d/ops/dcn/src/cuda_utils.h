#ifndef DCN_OPS_CUDA_UTILS_H
#define DCN_OPS_CUDA_UTILS_H

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define assert_expr(x) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, ""); }
#define assert_expr_msg(x, msg) if(!(x)) { assert_fail(#x, __FILE__, __LINE__, msg); }
inline void assert_fail(const char *str, const char *file, int line, std::string msg) {
    std::cerr << "Assertion failed " << str << " " << file << ":" << line << " " << msg << std::endl;
    abort();
}


#endif //DCN_OPS_CUDA_UTILS_H