#pragma once

#include "exo_gemm.h"

void exo_sgemm(const char transpose, const int m, const int n, const int k,
                const float *alpha, const float *beta,
                const float *A, const float *B, float *C) {

    if (*alpha==1.0 && *beta==1.0) {
        if (n<=32) {
            exo_sgemm_notranspose_noalpha_nobeta_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=64) {
            exo_sgemm_notranspose_noalpha_nobeta_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=128) {
            exo_sgemm_notranspose_noalpha_nobeta_128_128(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=256) {
            exo_sgemm_notranspose_noalpha_nobeta_256_256(nullptr, m, n, k, alpha, beta, A, B, C);
        } else {
            exo_sgemm_notranspose_noalpha_nobeta_main(nullptr, m, n, k, alpha, beta, A, B, C);
        }
    } else if (*alpha==0.0 && *beta==1.0) {
        return;
    } else if (*alpha==0.0 && *beta!=1.0) {
        if (n<=32) {
            exo_sgemm_alphazero_beta_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=64) {
            exo_sgemm_alphazero_beta_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=128) {
            exo_sgemm_alphazero_beta_128_128(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=256) {
            exo_sgemm_alphazero_beta_256_256(nullptr, m, n, k, alpha, beta, A, B, C);
        } else {
            exo_sgemm_alphazero_beta_main(nullptr, m, n, k, alpha, beta, A, B, C);
        }
    } else if (*alpha!=1.0 && *beta==1.0) {
        if (n<=32) {
            exo_sgemm_notranspose_alpha_nobeta_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=64) {
            exo_sgemm_notranspose_alpha_nobeta_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=128) {
            exo_sgemm_notranspose_alpha_nobeta_128_128(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=256) {
            exo_sgemm_notranspose_alpha_nobeta_256_256(nullptr, m, n, k, alpha, beta, A, B, C);
        } else {
            exo_sgemm_notranspose_alpha_nobeta_main(nullptr, m, n, k, alpha, beta, A, B, C);
        }
    } else {
        if (n<=32) {
            exo_sgemm_notranspose_alpha_beta_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=64) {
            exo_sgemm_notranspose_alpha_beta_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=128) {
            exo_sgemm_notranspose_alpha_beta_128_128(nullptr, m, n, k, alpha, beta, A, B, C);
        } else if (n<=256) {
            exo_sgemm_notranspose_alpha_beta_256_256(nullptr, m, n, k, alpha, beta, A, B, C);
        } else {
            exo_sgemm_notranspose_alpha_beta_main(nullptr, m, n, k, alpha, beta, A, B, C);
        }
    }

}