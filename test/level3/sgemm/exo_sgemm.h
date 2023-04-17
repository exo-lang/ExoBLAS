#pragma once

#include <cblas.h>
#include <stdio.h>

#include "exo_gemm.h"

void exo_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa,
               const enum CBLAS_TRANSPOSE transb, const int m, const int n,
               const int k, const float *alpha, const float *beta,
               const float *A, const float *B, float *C) {
  exo_sgemm_notranspose_noalpha_nobeta_main(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
}
