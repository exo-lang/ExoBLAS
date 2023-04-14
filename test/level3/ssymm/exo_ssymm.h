#pragma once

#include <cblas.h>

#include "exo_symm.h"

void exo_ssymm(const enum CBLAS_ORDER order, const CBLAS_SIDE side,
               const CBLAS_UPLO uplo, const int m, const int n,
               const float *alpha, const float *A, const int lda,
               const float *B, const int ldb, const float *beta, float *C,
               const int ldc) {
  exo_ssymm_lower_left_noalpha_nobeta_main(nullptr, m, n, m, A, B, C);
}
