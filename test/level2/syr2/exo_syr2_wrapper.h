#pragma once

#include "exo_syr2.h"

#define exo_syr2(type, prefix, exo_type)                                     \
  void exo_##prefix##syr2(                                                   \
      const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, \
      const type alpha, const type *X, const int incX, const type *Y,        \
      const int incY, type *A, const int lda) {                              \
    if (alpha == 0.0) {                                                      \
      return;                                                                \
    }                                                                        \
    if (incX < 0) {                                                          \
      X = X + (1 - N) * incX;                                                \
    }                                                                        \
    if (incY < 0) {                                                          \
      Y = Y + (1 - N) * incY;                                                \
    }                                                                        \
    if (Uplo == CBLAS_UPLO::CblasUpper) {                                    \
      if (incX == 1 && incY == 1) {                                          \
        exo_##prefix##syr2_rm_u_stride_1(                                    \
            nullptr, N, &alpha,                                              \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_2##exo_type{.data = A, .strides = {lda, 1}});            \
      } else {                                                               \
        exo_##prefix##syr2_rm_u_stride_any(                                  \
            nullptr, N, &alpha,                                              \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_2##exo_type{.data = A, .strides = {lda, 1}});            \
      }                                                                      \
    } else {                                                                 \
      if (incX == 1 && incY == 1) {                                          \
        exo_##prefix##syr2_rm_l_stride_1(                                    \
            nullptr, N, &alpha,                                              \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_2##exo_type{.data = A, .strides = {lda, 1}});            \
      } else {                                                               \
        exo_##prefix##syr2_rm_l_stride_any(                                  \
            nullptr, N, &alpha,                                              \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_1##exo_type##c{.data = Y, .strides = {incY}},            \
            exo_win_2##exo_type{.data = A, .strides = {lda, 1}});            \
      }                                                                      \
    }                                                                        \
  }

exo_syr2(float, s, f32);
exo_syr2(double, d, f64);
