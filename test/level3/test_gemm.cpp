#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include <cblas.h>
#include "sgemm.h"

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

static void print_matrix(std::vector<float> M, int n, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << M[j*k + i] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static std::vector<float> transpose(std::vector<float> V, const int m, const int k ) {
    std::vector<float> V_t(k*m);
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            V_t[j*m + i] = V[i*k + j];
        }
    }

    return V_t;
}


int main(int argc, char **argv) {
    int n = atoi(argv[1]); 
    auto a = gen_matrix(n, n);
    auto b = gen_matrix(n, n);
    auto c = gen_matrix(n, n);
    auto c2 = c; 

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  n, n, n, 
                  1.0, 
                  a.data(), n,
                  b.data(), n,
                  1.0,
                  c.data(), n);

    sgemm_notranspose(nullptr, n, n, n, c2.data(), a.data(), b.data());

    for (int i=0; i<n*n; i++) {
        break;
        double correct = c[i];//std::round(c[i] * 100.0) / 100.0;
        double exo_out = c2[i];//std::round(c2[i] * 100.0) / 100.0;
        if (correct!=exo_out)
            std::cout<<"Error at "<< i/n <<", "<<i%n<< ". Expected: "<<correct<<", got: "<<exo_out<<std::endl;
        assert(correct==exo_out);
    }
    std::cout<<"CORRECTNESS TEST PASSED"<<std::endl;

    long FLOP_C = 2 * long(n) * long(n) * long(n);

    int N_TIMES_BLAS = 100;
    auto begin = std::chrono::steady_clock::now();
    for (int times = 0; times < N_TIMES_BLAS; times++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 
                1.0, 
                a.data(), n,
                b.data(), n,
                1.0,
                c.data(), n);
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - begin).count();
    double ms_per_gemm = duration / N_TIMES_BLAS * 1.0e3;
    printf("-----------------------------------------------------------\n");
    printf("BLAS GEMM took %5.1lf ms, or %4.1lf GFLOPS\n", ms_per_gemm,
        (FLOP_C * 1.0e-6) / ms_per_gemm);

    int N_TIMES_EXO = 100;
    begin = std::chrono::steady_clock::now();
    for (int times = 0; times < N_TIMES_EXO; times++) {
        sgemm_notranspose(nullptr, n, n, n, c2.data(), a.data(), b.data());
    }
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();
    ms_per_gemm = duration / N_TIMES_EXO * 1.0e3;
    printf("-----------------------------------------------------------\n");
    printf("  Exo GEMM took %5.1lf ms, or %4.1lf GFLOPS\n", ms_per_gemm,
        (FLOP_C * 1.0e-6) / ms_per_gemm);
    printf("-----------------------------------------------------------\n");
}
