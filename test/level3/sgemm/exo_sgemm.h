#pragma once

#include "exo_gemm.h"

void exo_sgemm(const char transpose, const int m, const int n, const int k,
                const float *alpha, const float *beta,
                const float *A, const float *B, float *C) {

    if (*alpha==1.0 && *beta==1.0) {
        exo_gemm_notranspose_noalpha_nobeta(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (*alpha==0.0 && *beta==1.0) {
        exo_gemm_alphazero_nobeta(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (*alpha==0.0 && *beta!=1.0) {
        exo_gemm_alphazero_beta(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (*alpha!=1.0 && *beta==1.0) {
        exo_gemm_notranspose_alpha_nobeta(nullptr, m, n, k, alpha, beta, A, B, C);
    } else {
        exo_gemm_notranspose_alpha_beta(nullptr, m, n, k, alpha, beta, A, B, C);
    }

}