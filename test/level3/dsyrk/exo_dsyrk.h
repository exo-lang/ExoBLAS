#pragma once

#include "exo_syrk.h"

void exo_dsyrk(const char uplo, const char transpose, 
                const int n, const int k,
                const double *alpha,
                const double *A1, const double *A2,
                const double *beta, double *C){
    
    //TODO: other cases
    exo_dsyrk_lower_notranspose(nullptr, n, k, alpha, A1, A2, beta, C);

}