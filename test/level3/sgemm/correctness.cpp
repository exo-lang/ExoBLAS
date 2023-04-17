#include <cblas.h>

#include <iostream>
#include <vector>

#include "correctness_helpers.h"
#include "exo_sgemm.h"
#include "generate_buffer.h"

void test_sgemm(const enum CBLAS_TRANSPOSE transa,
                const enum CBLAS_TRANSPOSE transb, const int n, const int m,
                const int k, const float alpha, const float beta) {
  std::cout << "Running sgemm test: N = " << n << ", alpha = " << alpha
            << ", beta = " << beta << ", trasnsa = " << transa
            << ", transb = " << transb << "..." << std::endl;
  auto a = AlignedBuffer2D<float>(m, k);
  auto b = AlignedBuffer2D<float>(k, n);
  auto c = AlignedBuffer2D<float>(m, n);
  auto c2 = c;

  cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, alpha, a.data(), m,
              b.data(), k, beta, c.data(), m);

  exo_sgemm(CblasRowMajor, transa, transb, m, n, k, &alpha, &beta, a.data(),
            b.data(), c2.data());

  double epsilon = 0.01;
  for (int i = 0; i < m * n; i++) {
    double correct = c[i];
    double exo_out = c2[i];
    if (!check_relative_error_okay(correct, exo_out, epsilon)) {
      std::cout << "Error at " << i / n << ", " << i % n
                << ". Expected: " << correct << ", got: " << exo_out
                << std::endl;
      exit(1);
    }
  }

  std::cout << "Passed!" << std::endl;
}

int main() {
  std::vector<int> dims{32, 64, 256, 512, 2048};
  std::vector<CBLAS_TRANSPOSE> transas{ CblasNoTrans};
  std::vector<CBLAS_TRANSPOSE> transbs{ CblasNoTrans};
  std::vector<float> alphas{1.0};
  std::vector<float> betas{1.0};

  for (auto const n : dims) {
    for (auto const transa : transas) {
      for (auto const transb : transbs) {
        for (auto const alpha : alphas) {
          for (auto const beta : betas) {
            test_sgemm(transa, transb, n, n, n, alpha, beta);
          }
        }
      }
    }
  }
}
