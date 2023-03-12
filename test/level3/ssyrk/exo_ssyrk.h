#pragma once

#include "exo_syrk.h"

void exo_ssyrk(const char uplo, const char transpose, 
                const int n, const int k,
                const float *alpha,
                const float *A1, const float *A2,
                const float *beta, float *C){
    
    //TODO: other cases
    exo_ssyrk_lower_notranspose(nullptr, n, k, alpha, A1, A2, beta, C);

}