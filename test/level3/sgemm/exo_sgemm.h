#pragma once

#include <cblas.h>

#include "error.h"
#include "exo_gemm.h"

void exo_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
               const int K, const float alpha, const float *A, const int lda,
               const float *B, const int ldb, const float beta, float *C,
               const int ldc) {
  if (Order != CBLAS_ORDER::CblasRowMajor) {
    throw UnsupportedParameterException("sgemm::Order must be Row Major");
  }
  if (TransA != CBLAS_TRANSPOSE::CblasNoTrans) {
    throw UnsupportedParameterException("sgemm::TransA must be nonTrans");
  }
  if (TransB != CBLAS_TRANSPOSE::CblasNoTrans) {
    throw UnsupportedParameterException("sgemm::TransB must be nonTrans");
  }
  if (alpha != 1.0) {
    throw UnsupportedParameterException("sgemm::alpha must be 1.0");
  }
  if (beta != 1.0) {
    throw UnsupportedParameterException("sgemm::beta must be 1.0");
  }
  if (lda != K) {
    throw UnsupportedParameterException("sgemm::lda must be K");
  }
  if (ldb != N) {
    throw UnsupportedParameterException("sgemm::ldb must be N");
  }
  if (ldc != N) {
    throw UnsupportedParameterException("sgemm::ldc must be N");
  }
  exo_sgemm_matmul_stride_any(nullptr, M, N, K,
                              exo_win_2f32c{.data = A, .strides = {lda, 1}},
                              exo_win_2f32c{.data = B, .strides = {ldb, 1}},
                              exo_win_2f32{.data = C, .strides{ldc, 1}});
}
