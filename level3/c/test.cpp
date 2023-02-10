#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>

#include <chrono>


#include <cblas.h>
#include "syrk.c"
//#include "output.c"

static std::vector<float> gen_matrix(long m, long n, float v) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  //std::generate(std::begin(mat), std::end(mat), [&]() { return v; });
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

static std::vector<float> gen_matrix_symm(long m, long n) {
    std::vector<float> mat(m * n);
   mat={2.47 ,4.38 ,7.89 ,8.79 ,8.34 ,8.97 ,6.73 ,7.57 ,7.85 ,8.90 ,2.60 ,4.49 ,2.23 ,4.95 ,3.87 ,5.13,
        4.38 ,3.64 ,2.92 ,8.35 ,8.65 ,8.99 ,7.35 ,1.43 ,0.44 ,7.79 ,2.52 ,2.61 ,3.12 ,8.02 ,4.00 ,7.80,
        7.89 ,2.92 ,8.69 ,3.91 ,7.22 ,4.07 ,0.69 ,3.32 ,8.24 ,5.89 ,6.66 ,8.29 ,6.95 ,7.71 ,7.38 ,1.03,
        8.79 ,8.35 ,3.91 ,3.83 ,5.33 ,0.92 ,2.36 ,1.00 ,7.99 ,6.95 ,8.11 ,8.18 ,8.17 ,6.67 ,8.69 ,1.54,
        8.34 ,8.65 ,7.22 ,5.33 ,4.46 ,0.03 ,5.03 ,6.55 ,0.37 ,7.36 ,7.15 ,6.39 ,2.61 ,1.35 ,1.66 ,2.14,
        8.97 ,8.99 ,4.07 ,0.92 ,0.03 ,0.15 ,2.31 ,3.45 ,3.10 ,6.88 ,4.02 ,2.77 ,5.47 ,5.62 ,3.96 ,8.54,
        6.73 ,7.35 ,0.69 ,2.36 ,5.03 ,2.31 ,0.86 ,3.07 ,4.00 ,8.88 ,6.37 ,5.70 ,4.85 ,7.97 ,6.16 ,5.06,
        7.57 ,1.43 ,3.32 ,1.00 ,6.55 ,3.45 ,3.07 ,0.89 ,0.21 ,6.69 ,0.05 ,4.86 ,0.04 ,5.11 ,5.68 ,3.27,
        7.85 ,0.44 ,8.24 ,7.99 ,0.37 ,3.10 ,4.00 ,0.21 ,4.49 ,5.52 ,6.71 ,4.75 ,2.97 ,7.05 ,5.41 ,2.38,
        8.90 ,7.79 ,5.89 ,6.95 ,7.36 ,6.88 ,8.88 ,6.69 ,5.52 ,1.48 ,6.86 ,1.18 ,5.73 ,5.42 ,0.69 ,1.01,
        2.60 ,2.52 ,6.66 ,8.11 ,7.15 ,4.02 ,6.37 ,0.05 ,6.71 ,6.86 ,7.02 ,4.18 ,4.59 ,5.11 ,6.00 ,2.83,
        4.49 ,2.61 ,8.29 ,8.18 ,6.39 ,2.77 ,5.70 ,4.86 ,4.75 ,1.18 ,4.18 ,3.49 ,4.77 ,6.09 ,5.09 ,2.24,
        2.23 ,3.12 ,6.95 ,8.17 ,2.61 ,5.47 ,4.85 ,0.04 ,2.97 ,5.73 ,4.59 ,4.77 ,3.04 ,7.34 ,5.85 ,1.01,
        4.95 ,8.02 ,7.71 ,6.67 ,1.35 ,5.62 ,7.97 ,5.11 ,7.05 ,5.42 ,5.11 ,6.09 ,7.34 ,2.39 ,8.05 ,0.32,
        3.87 ,4.00 ,7.38 ,8.69 ,1.66 ,3.96 ,6.16 ,5.68 ,5.41 ,0.69 ,6.00 ,5.09 ,5.85 ,8.05 ,8.24 ,7.70,
        5.13 ,7.80 ,1.03 ,1.54 ,2.14 ,8.54 ,5.06 ,3.27 ,2.38 ,1.01 ,2.83 ,2.24 ,1.01 ,0.32 ,7.70 ,3.54
                
        };
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

    auto a = gen_matrix(n, n, 2.0);
    auto a2 = transpose(a, n, n);
    auto c = gen_matrix_symm(n, n);
    auto c2 = c; 

    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, n, // M N K
                1.0, // alpha
                a.data(),
                n, // M
                1.0,
                c.data(),
                n  // M
                );
   /* cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, 10, // M N K
                1.0, // alpha
                a.data(),
                n, // M
               a.data(),
                n, // K
                1.0, // beta
                c.data(),
                n  // M
                );*/

    std::cout << "Correct output:" << std::endl;
    print_matrix(c, n, n);

    std::cout << "Matrix C:" << std::endl;
    print_matrix(c2, n, n);
    std::cout << "Matrix A:" << std::endl;
    print_matrix(a, n, n);

    std::cout << "Actual Output:" << std::endl;
    //SYRK2(nullptr, n, n, a.data(), transpose(a, n, n).data(), c2.data());
    SYRK(nullptr, n, n, a.data(), c2.data());
    print_matrix(c2, n, n);
    //SGEMM3(NULL, n, n, 16, a.data(), a.data(), c2.data());
    //SYRK(NULL, n, n, a.data(), a.data(), c2.data());
    //sgemm_exo(NULL, n, n, n, a.data(), b.data(), c2.data());
    /*auto A_micro = gen_matrix(4, 4, 2.0);
    auto C_micro = gen_matrix(4, 4, 2.0);
    auto C_micro_2 = C_micro;

    microkernel_two(nullptr, A_micro.data(), A_micro.data(), C_micro_2.data());
    std::cout << "Correct output:" << std::endl;
    print_matrix(C_micro_2, 4, 4);

    std::cout << "Matrix C:" << std::endl;
    print_matrix(C_micro, 4, 4);
    std::cout << "Matrix A:" << std::endl;
    print_matrix(A_micro, 4, 4);

    std::cout << "Actual Output:" << std::endl;
    neon_microkernel_two_4(nullptr, A_micro.data(), transpose(A_micro, 4, 4).data(), C_micro.data());
    print_matrix(C_micro, 4, 4);*/

    //print_matrix(c2, n, n);
}