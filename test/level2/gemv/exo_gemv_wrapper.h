#pragma once

#include <cblas.h>

#include "exo_gemv.h"

#define exo_gemv(type, prefix, exo_type)                                  \
  void exo_##prefix##gemv(const enum CBLAS_ORDER order,                   \
                          const enum CBLAS_TRANSPOSE TransA, const int M, \
                          const int N, const type alpha, const type *A,   \
                          const int lda, const type *X, const int incX,   \
                          const type beta, type *Y, const int incY) {     \
    if (order != CBLAS_ORDER::CblasRowMajor) {                            \
      throw "Unsupported Exception";                                      \
    }                                                                     \
                                                                          \
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {                        \
      if (incX == 1 && incY == 1) {                                       \
        exo_##prefix##gemv_rm_nt_stride_1(                                \
            nullptr, M, N, &alpha, &beta,                                 \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},       \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});           \
      } else {                                                            \
        if (incX < 0) {                                                   \
          X = X + (1 - N) * incX;                                         \
        }                                                                 \
        if (incY < 0) {                                                   \
          Y = Y + (1 - M) * incY;                                         \
        }                                                                 \
        exo_##prefix##gemv_rm_nt_stride_any(                              \
            nullptr, M, N, &alpha, &beta,                                 \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},       \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});           \
      }                                                                   \
    } else {                                                              \
      if (incX == 1 && incY == 1) {                                       \
        exo_##prefix##gemv_rm_t_stride_1(                                 \
            nullptr, N, M, &alpha, &beta,                                 \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},       \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});           \
      } else {                                                            \
        if (incX < 0) {                                                   \
          X = X + (1 - M) * incX;                                         \
        }                                                                 \
        if (incY < 0) {                                                   \
          Y = Y + (1 - N) * incY;                                         \
        }                                                                 \
        exo_##prefix##gemv_rm_t_stride_any(                               \
            nullptr, N, M, &alpha, &beta,                                 \
            exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},       \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});           \
      }                                                                   \
    }                                                                     \
  }

exo_gemv(float, s, f32);
exo_gemv(double, d, f64);
