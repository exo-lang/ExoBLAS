#pragma once

#include "exo_syr2k.h"

#include <cblas.h>


void exo_ssyr2k(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
                const int N, const int K,
                const float * alpha,
                const float * A, const int lda,
                const float * B, const int ldb,
                const float * beta,
                float * C, const int ldc) {
    
    exo_ssyr2k_lower_notranspose_noalpha_main(nullptr, N, K, A, alpha, B, beta, C);

}