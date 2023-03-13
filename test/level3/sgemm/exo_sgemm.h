#pragma once

#include "exo_gemm.h"

void exo_sgemm(const char transpose, const int m, const int n, const int k,
                const float *alpha, const float *beta,
                const float *A, const float *B, float *C) {

    if (*alpha==1.0 && *beta==1.0) {
        switch(n){
            case 32:
                exo_gemm_notranspose_noalpha_nobeta_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
                break;
            case 64:
                exo_gemm_notranspose_noalpha_nobeta_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
                break;
            case 128:
                exo_gemm_notranspose_noalpha_nobeta_128_128(nullptr, m, n, k, alpha, beta, A, B, C);
                break;
            case 256:
                exo_gemm_notranspose_noalpha_nobeta_256_256(nullptr, m, n, k, alpha, beta, A, B, C);
                break;
            default:
                exo_gemm_notranspose_noalpha_nobeta(nullptr, m, n, k, alpha, beta, A, B, C);
                break;
        }
            
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