#pragma once

#include <cblas.h>

#include "error.h"
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
    if (beta != 1.0) {                                                        \
      throw UnsupportedParameterException("symm::beta must be 1.0");          \
    }                                                                         \
    if (Uplo == CBLAS_UPLO::CblasLower) {                                     \
      if (Side == CBLAS_SIDE::CblasLeft) {                                    \
        exo_##prefix##symm_rm_ll_stride_1(                                    \
            nullptr, M, N, &alpha,                                            \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},           \
            exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},           \
            exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                \
      } else {                                                                \
        exo_##prefix##symm_rm_rl_stride_1(                                    \
            nullptr, M, N, &alpha,                                            \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},           \
            exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},           \
            exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                \
      }                                                                       \
    } else {                                                                  \
      if (Side == CBLAS_SIDE::CblasLeft) {                                    \
        exo_##prefix##symm_rm_lu_stride_1(                                    \
            nullptr, M, N, &alpha,                                            \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},           \
            exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},           \
            exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                \
      } else {                                                                \
        exo_##prefix##symm_rm_ru_stride_1(                                    \
            nullptr, M, N, &alpha,                                            \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},           \
            exo_win_2##exo_type##c{.data = B, .strides = {ldb, 1}},           \
            exo_win_2##exo_type{.data = C, .strides{ldc, 1}});                \
      }                                                                       \
    }                                                                         \
  }

exo_symm(float, s, f32);
exo_symm(double, d, f64);
