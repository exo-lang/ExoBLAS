#pragma once

#include "exo_trmm.h"
#include <cblas.h>

void exo_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb) {
    if (Order != CBLAS_ORDER::CblasRowMajor) {
        throw "Unsupported Exception";
    }
    if (Side == CBLAS_SIDE::CblasLeft) {
        if (Uplo == CBLAS_UPLO::CblasUpper) {
            if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
                exo_strmm_row_major_Left_Upper_NonTrans(nullptr, M, N, &alpha,
                    exo_win_2f32c{.data=A, .strides={lda, 1}},
                    exo_win_2f32{.data=B, .strides={ldb, 1}},
                    Diag == CBLAS_DIAG::CblasUnit);
            } else {
                exo_strmm_row_major_Left_Upper_Trans(nullptr, M, N, &alpha,
                    exo_win_2f32c{.data=A, .strides={lda, 1}},
                    exo_win_2f32{.data=B, .strides={ldb, 1}},
                    Diag == CBLAS_DIAG::CblasUnit
                );
            }
        } else {
            if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
               exo_strmm_row_major_Left_Lower_NonTrans(nullptr, M, N, &alpha,
                    exo_win_2f32c{.data=A, .strides={lda, 1}},
                    exo_win_2f32{.data=B, .strides={ldb, 1}},
                    Diag == CBLAS_DIAG::CblasUnit);
            } else {
                exo_strmm_row_major_Left_Lower_Trans(nullptr, M, N, &alpha,
                    exo_win_2f32c{.data=A, .strides={lda, 1}},
                    exo_win_2f32{.data=B, .strides={ldb, 1}},
                    Diag == CBLAS_DIAG::CblasUnit);
            }
        }
    } else {
        if (Uplo == CBLAS_UPLO::CblasUpper) {
            if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
                
            } else {
                
            }
        } else {
            if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
               
            } else {
                
            }
        }
    }
}
