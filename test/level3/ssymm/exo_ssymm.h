#pragma once

#include <cblas.h>

#include "exo_symm.h"

void exo_ssymm(const enum CBLAS_ORDER order, const CBLAS_SIDE side,
               const CBLAS_UPLO uplo, const int m, const int n,
               const float *alpha, const float *A, const int lda,
               const float *B, const int ldb, const float *beta, float *C,
               const int ldc) {
  if (n <= 48) {
    exo_ssymm_lower_left_noalpha_nobeta_48_48_48(nullptr, m, n, m, A, B, C);
  } else if (n <= 96) {
    exo_ssymm_lower_left_noalpha_nobeta_96_96_96(nullptr, m, n, m, A, B, C);
  } else if (n <= 192) {
    exo_ssymm_lower_left_noalpha_nobeta_192_192_192(nullptr, m, n, m, A, B, C);
  } else if (n <= 384) {
    exo_ssymm_lower_left_noalpha_nobeta_384_384_384(nullptr, m, n, m, A, B, C);
  } else if (n <= 480) {
    exo_ssymm_lower_left_noalpha_nobeta_480_240_480(nullptr, m, n, m, A, B, C);
  } else {
    exo_ssymm_lower_left_noalpha_nobeta_960_240_480(nullptr, m, n, m, A, B, C);
  }
}
