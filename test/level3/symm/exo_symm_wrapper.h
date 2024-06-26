#pragma once

#include <cblas.h>

#include "error.h"
#include "exo_mscal.h"
#include "exo_symm.h"

#define exo_symm(type, prefix, exo_type)                                      \
  void exo_##prefix##symm(                                                    \
      const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,               \
      const enum CBLAS_UPLO Uplo, const int M, const int N, const type alpha, \
      const type *A, const int lda, const type *B, const int ldb,             \
      const type beta, type *C, const int ldc) {                              \
    if (Order != CBLAS_ORDER::CblasRowMajor) {                                \
      throw UnsupportedParameterException("symm::Order must be Row Major");   \
    }                                                                         \
    exo_##prefix##mscal_rm_stride_1(                                          \
        nullptr, M, N, &beta,                                                 \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                    \
    exo_##prefix##symm_rm_stride_1(                                           \
        nullptr, Side, Uplo, M, N, &alpha,                                    \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},               \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},               \
        exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},               \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                    \
  }

exo_symm(float, s, f32);
exo_symm(double, d, f64);
