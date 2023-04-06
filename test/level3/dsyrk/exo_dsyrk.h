#pragma once

#include "exo_syrk.h"

#include <cblas.h>

void exo_dsyrk_lower_notranspose(const int n, const int k,
                                 const double *alpha,
                                 const double *A1, const double *A2,
                                 const double *beta, double *C) {
    if (*alpha==1.0 && *beta==1.0) {
        exo_dsyrk_lower_notranspose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha==0.0 && *beta==1.0) {
        return;
    } else if (*alpha==0.0 && *beta!=1.0) {
        exo_dsyrk_lower_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha!=1.0 && *beta==1.0) {
        exo_dsyrk_lower_notranspose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else {
        exo_dsyrk_lower_notranspose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    }
}

void exo_dsyrk_lower_transpose(const int n, const int k,
                                 const double *alpha,
                                 const double *A1, const double *A2,
                                 const double *beta, double *C) {
    if (*alpha==1.0 && *beta==1.0) {
        exo_dsyrk_lower_transpose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha==0.0 && *beta==1.0) {
        return;
    } else if (*alpha==0.0 && *beta!=1.0) {
        exo_dsyrk_lower_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha!=1.0 && *beta==1.0) {
        exo_dsyrk_lower_transpose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else {
        exo_dsyrk_lower_transpose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    }
}

void exo_dsyrk_upper_notranspose(const int n, const int k,
                                 const double *alpha,
                                 const double *A1, const double *A2,
                                 const double *beta, double *C) {
    if (*alpha==1.0 && *beta==1.0) {
        exo_dsyrk_upper_notranspose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha==0.0 && *beta==1.0) {
        return;
    } else if (*alpha==0.0 && *beta!=1.0) {
        exo_dsyrk_upper_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha!=1.0 && *beta==1.0) {
        exo_dsyrk_upper_notranspose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else {
        exo_dsyrk_upper_notranspose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    }
}

void exo_dsyrk_upper_transpose(const int n, const int k,
                                 const double *alpha,
                                 const double *A1, const double *A2,
                                 const double *beta, double *C) {
    if (*alpha==1.0 && *beta==1.0) {
        exo_dsyrk_upper_transpose_noalpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha==0.0 && *beta==1.0) {
        return;
    } else if (*alpha==0.0 && *beta!=1.0) {
        exo_dsyrk_upper_alphazero_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else if (*alpha!=1.0 && *beta==1.0) {
        exo_dsyrk_upper_transpose_alpha_nobeta(nullptr, n, k, alpha, A1, A2, beta, C);
    } else {
        exo_dsyrk_upper_transpose_alpha_beta(nullptr, n, k, alpha, A1, A2, beta, C);
    }
}

void exo_dsyrk(const enum CBLAS_ORDER order,
                const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transpose, 
                const int n, const int k,
                const double *alpha,
                const double *A1, const double *A2,
                const double *beta, double *C){
    
    //TODO: other cases
    if (uplo==CblasLower) {
        if (transpose==CblasNoTrans) {
            exo_dsyrk_lower_notranspose(n, k, alpha, A1, A2, beta, C);
        } else {
            exo_dsyrk_lower_transpose(n, k, alpha, A1, A2, beta, C);
        }
    } else {
        if (transpose==CblasNoTrans) {
            exo_dsyrk_upper_notranspose(n, k, alpha, A1, A2, beta, C);
        } else {
            exo_dsyrk_upper_transpose(n, k, alpha, A1, A2, beta, C);
        }
    }

}