#pragma once

#include <cblas.h>

#include "error.h"
#include "exo_mscal.h"
#include "exo_syrk.h"

#define exo_syrk(type, prefix, exo_type)                                    \
  void exo_##prefix##syrk(                                                  \
      const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,             \
      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,           \
      const type alpha, const type *A, const int lda, const type beta,      \
      type *C, const int ldc) {                                             \
    if (Order != CBLAS_ORDER::CblasRowMajor) {                              \
      throw UnsupportedParameterException("syrk::Order must be Row Major"); \
    }                                                                       \
    exo_##prefix##trmscal_rm_stride_1(                                      \
        nullptr, Uplo, N, &beta,                                            \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                  \
    exo_##prefix##syrk_rm_stride_1(                                         \
        nullptr, Uplo, Trans, N, K, &alpha,                                 \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},             \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},             \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},             \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},             \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                  \
  }

exo_syrk(float, s, f32);
exo_syrk(double, d, f64);
