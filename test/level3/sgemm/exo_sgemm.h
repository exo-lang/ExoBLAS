#pragma once

#include <cblas.h>
#include <stdio.h>

#include "exo_gemm.h"

void exo_sgemm_notranspose(const int m, const int n, const int k,
                           const float *alpha, const float *beta,
                           const float *A, const float *B, float *C) {
  exo_sgemm_test(m, n, k, A, B, C);
}
