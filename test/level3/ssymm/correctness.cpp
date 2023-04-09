#include <vector>
#include <iostream>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_ssymm.h"

void test_ssymm(const enum CBLAS_SIDE side, const enum CBLAS_UPLO uplo,
                const int n, const int m,
                const float alpha, const float beta) {
    
    std::cout<<"Running ssymm test: N = "<<n<<", alpha = "<<alpha<<", beta = "<<beta<< ", side = "<<side<<", uplo = "<<uplo<<"..."<<std::endl;
    auto a = AlignedBuffer2D<float>(m, m, 64, 2.0);
    auto b = AlignedBuffer2D<float>(m, n);
    auto c = AlignedBuffer2D<float>(m, n);
    auto c2 = c; 

    cblas_ssymm(CblasRowMajor, side, uplo,
                  m, n, 
                  alpha, 
                  a.data(), m,
                  b.data(), m,
                  beta,
                  c.data(), m);

    exo_ssymm(CblasRowMajor, side, uplo, m, n, &alpha, a.data(), m, b.data(), m, &beta, c2.data(), m);

    double epsilon = 0.01;
    for (int i=0; i<m*n; i++) {
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
    
    std::vector<int> dims {32, 64, 256, 257};
    std::vector<CBLAS_SIDE> sides {CblasLeft};
    std::vector<CBLAS_UPLO> uplos {CblasLower};
    std::vector<float> alphas {1.0};
    std::vector<float> betas {1.0};

    for (auto const n : dims) {
        for (auto const side : sides) {
            for (auto const uplo : uplos) {
                for (auto const alpha : alphas) {
                    for (auto const beta : betas) {
                        test_ssymm(side, uplo, n, n, alpha, beta);
                    }
                }
            }
        }
    }

}