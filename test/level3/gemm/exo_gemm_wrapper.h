#pragma once

#include <cblas.h>

#include "error.h"
#include "exo_gemm.h"
#include "exo_mscal.h"

#define exo_gemm(type, prefix, exo_type)                                       \
  void exo_##prefix##gemm(                                                     \
      const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,         \
      const enum CBLAS_TRANSPOSE TransB, const int M, const int N,             \
      const int K, const type alpha, const type *A, const int lda,             \
      const type *B, const int ldb, const type beta, type *C, const int ldc) { \
    if (Order != CBLAS_ORDER::CblasRowMajor) {                                 \
      throw UnsupportedParameterException("sgemm::Order must be Row Major");   \
    }                                                                          \
    exo_##prefix##mscal_rm_stride_1(                                           \
        nullptr, M, N, &beta,                                                  \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                     \
    exo_##prefix##gemm_stride_1(                                               \
        nullptr, TransA, TransB, M, N, K, &alpha,                              \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},                \
        exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},                \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},                \
        exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},                \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                     \
  }

exo_gemm(float, s, f32);
exo_gemm(double, d, f64);
