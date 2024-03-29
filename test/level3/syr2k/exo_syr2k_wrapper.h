#pragma once

#include <cblas.h>

#include "error.h"
#include "exo_syr2k.h"

#define exo_syr2k(type, prefix, exo_type)                                    \
  void exo_##prefix##syr2k(                                                  \
      const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,              \
      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,            \
      const type alpha, const type *A, const int lda, const type *B,         \
      const int ldb, const type beta, type *C, const int ldc) {              \
    if (Order != CBLAS_ORDER::CblasRowMajor) {                               \
      throw UnsupportedParameterException("syr2k::Order must be Row Major"); \
    }                                                                        \
    if (beta != 1.0) {                                                       \
      throw UnsupportedParameterException("syr2k::beta must be 1.0");        \
    }                                                                        \
    exo_##prefix##syr2k_rm_stride_1(                                         \
        nullptr, Uplo, Trans, N, K, &alpha,                                  \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},              \
        exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},              \
        exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},              \
        exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},              \
        exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                   \
  }

exo_syr2k(float, s, f32);
exo_syr2k(double, d, f64);
