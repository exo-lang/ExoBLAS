#include <vector>
#include <iostream>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_sgemm.h"

void test_sgemm(const char transpose,
                const int n, const int m, const int k,
                const float alpha, const float beta) {
    
    std::cout<<"Running sgemm test: N = "<<n<<", alpha = "<<alpha<<", beta = "<<beta<<"..."<<std::endl;
    auto a = AlignedBuffer2D<float>(m, k);
    auto b = AlignedBuffer2D<float>(k, n);
    auto c = AlignedBuffer2D<float>(m, n);
    auto c2 = c; 

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m, n, k, 
                  alpha, 
                  a.data(), m,
                  b.data(), k,
                  beta,
                  c.data(), m);

    exo_sgemm('N', m, n, k, &alpha, &beta, a.data(), b.data(), c2.data());

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

    for (auto const n : dims) {
        test_sgemm('N', n, n, n, 1.0, 1.0);
        test_sgemm('N', n, n, n, 0.0, 1.0);
        test_sgemm('N', n, n, n, 1.0, 0.0);

        test_sgemm('N', n, n, n, 0.0, 0.0);

        test_sgemm('N', n, n, n, 1.0, 2.0);
        test_sgemm('N', n, n, n, 0.0, 2.0);

        test_sgemm('N', n, n, n, 2.0, 1.0);
        test_sgemm('N', n, n, n, 2.0, 0.0);

        test_sgemm('N', n, n, n, 2.0, 2.0);
    }

}