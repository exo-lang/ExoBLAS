#include <vector>
#include <iostream>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_ssyrk.h"

static std::vector<float> _transpose(float * V, const int m, const int k ) {
    std::vector<float> V_t(k*m);
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            V_t[j*m + i] = V[i*k + j];
        }
    }

    return V_t;
}

void test_ssyrk(const char uplo, const char transpose,
                const int n, const int k, 
                const float alpha, const float beta) {
    
    std::cout<<"Running syrk test: N = "<<n<<", alpha = "<<alpha<<", beta = "<<beta<<std::endl;
    auto a = AlignedBuffer2D<float>(n, k);
    auto a2 = _transpose(a.data(), n, n);
    auto c = AlignedBuffer2D<float>(n, k, 2.0f, 64);
    auto c2 = c;

    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, n, // M N
                1.0, // alpha
                a.data(),
                n, // M
                1.0,
                c.data(),
                n  // M
                );

    exo_ssyrk(uplo, transpose, n, n, &alpha, a.data(), a2.data(), &beta, c2.data());

    double epsilon = 0.01;
    for (int i=0; i<n*k; i++) {
        double correct = c[i];
        double exo_out = c2[i];
        if (!check_relative_error_okay(correct, exo_out, epsilon)) {
            std::cout<<"Error at "<< i/n <<", "<<i%n<< ". Expected: "<<correct<<", got: "<<exo_out<<std::endl;
            exit(1);
        }
    }

    std::cout<<"Passed!"<<std::endl;

}

int main() {
    
    std::vector<int> dims {32, 64, 256, 512, 513};

    for (auto const n : dims) {
        test_ssyrk('L', 'N', n, n, 1.0, 1.0);
    }

}