#pragma once

#include <cblas.h>

#include "exo_syrk.h"

void exo_ssyrk_lower_notranspose(const int n, const int k, const float *alpha,
                                 const float *A1, const float *A2,
                                 const float *beta, float *C) {
  if (*alpha == 1.0 && *beta == 1.0) {
    exo_ssyrk_lower_notranspose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2,
                                               beta, C);
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    exo_ssyrk_lower_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  } else if (*alpha != 1.0 && *beta == 1.0) {
    exo_ssyrk_lower_notranspose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                             C);
  } else {
    exo_ssyrk_lower_notranspose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta,
                                           C);
  }
}

void exo_ssyrk_lower_transpose(const int n, const int k, const float *alpha,
                               const float *A1, const float *A2,
                               const float *beta, float *C) {
  if (*alpha == 1.0 && *beta == 1.0) {
    exo_ssyrk_lower_transpose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                             C);
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    exo_ssyrk_lower_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  } else if (*alpha != 1.0 && *beta == 1.0) {
    exo_ssyrk_lower_transpose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                           C);
  } else {
    exo_ssyrk_lower_transpose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  }
}

void exo_ssyrk_upper_notranspose(const int n, const int k, const float *alpha,
                                 const float *A1, const float *A2,
                                 const float *beta, float *C) {
  if (*alpha == 1.0 && *beta == 1.0) {
    exo_ssyrk_upper_notranspose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2,
                                               beta, C);
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    exo_ssyrk_upper_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  } else if (*alpha != 1.0 && *beta == 1.0) {
    exo_ssyrk_upper_notranspose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                             C);
  } else {
    exo_ssyrk_upper_notranspose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta,
                                           C);
  }
}

void exo_ssyrk_upper_transpose(const int n, const int k, const float *alpha,
                               const float *A1, const float *A2,
                               const float *beta, float *C) {
  if (*alpha == 1.0 && *beta == 1.0) {
    exo_ssyrk_upper_transpose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                             C);
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    exo_ssyrk_upper_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  } else if (*alpha != 1.0 && *beta == 1.0) {
    exo_ssyrk_upper_transpose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta,
                                           C);
  } else {
    exo_ssyrk_upper_transpose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
  }
}

void exo_ssyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
               const enum CBLAS_TRANSPOSE transpose, const int n, const int k,
               const float *alpha, const float *A1, const float *A2,
               const float *beta, float *C) {
  // TODO: other cases
  if (uplo == CblasLower) {
    if (transpose == CblasNoTrans) {
      exo_ssyrk_lower_notranspose(n, k, alpha, A1, A2, beta, C);
    } else {
      exo_ssyrk_lower_transpose(n, k, alpha, A1, A2, beta, C);
    }
  } else {
    if (transpose == CblasNoTrans) {
      exo_ssyrk_upper_notranspose(n, k, alpha, A1, A2, beta, C);
    } else {
      exo_ssyrk_upper_transpose(n, k, alpha, A1, A2, beta, C);
    }
  }
}
