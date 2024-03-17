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
  if (beta != 1.0) {
    throw UnsupportedParameterException("sgemm::beta must be 1.0");
  }
  exo_sgemm_stride_1(nullptr, M, N, K, &alpha,
                     exo_win_2f32c{.data = A, .strides = {lda, 1}},
                     exo_win_2f32c{.data = B, .strides = {ldb, 1}},
                     exo_win_2f32{.data = C, .strides{ldc, 1}});
}
