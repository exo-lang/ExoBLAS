#include <cblas.h>

#include <iostream>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dsyrk.h"
#include "generate_buffer.h"

static std::vector<double> _transpose(double *V, const int m, const int k) {
  std::vector<double> V_t(k * m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      V_t[j * m + i] = V[i * k + j];
    }
  }

  return V_t;
}

void test_dsyrk(const enum CBLAS_UPLO uplo,
                const enum CBLAS_TRANSPOSE transpose, const int n, const int k,
                const double alpha, const double beta) {
  std::cout << "Running syrk test: N = " << n << ", alpha = " << alpha
            << ", beta = " << beta << " uplo = " << uplo
            << ", transpose = " << transpose << std::endl;
  auto a = AlignedBuffer2D<double>(n, k);
  auto a2 = a;
  auto c = AlignedBuffer2D<double>(n, k, 2.0f, 64);
  auto c2 = c;

  cblas_dsyrk(CblasRowMajor, uplo, transpose, n, n,  // M N
              alpha,                                 // alpha
              a.data(),
              n,  // M
              beta, c.data(),
              n  // M
  );
  if (uplo == CblasLower && transpose == CblasNoTrans && alpha == 1.0 &&
      beta == 1.0) {
    auto at = _transpose(a.data(), n, n);
    exo_dsyrk(CblasRowMajor, uplo, transpose, n, n, &alpha, a.data(), at.data(),
              &beta, c2.data());
  } else {
    exo_dsyrk(CblasRowMajor, uplo, transpose, n, n, &alpha, a.data(), a2.data(),
              &beta, c2.data());
  }

  double epsilon = 0.01;
  for (int i = 0; i < n * k; i++) {
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
  std::vector<int> dims{32, 64, 256, 513};
  std::vector<CBLAS_UPLO> uplos{CblasLower, CblasUpper};
  std::vector<CBLAS_TRANSPOSE> transposes{CblasNoTrans, CblasTrans};
  std::vector<double> alphas{0.0, 1.0, 2.0};
  std::vector<double> betas{0.0, 1.0, 2.0};

  for (auto const n : dims) {
    for (auto const uplo : uplos) {
      for (auto const transpose : transposes) {
        for (auto const alpha : alphas) {
          for (auto const beta : betas) {
            test_dsyrk(uplo, transpose, n, n, alpha, beta);
          }
        }
      }
    }
    // test_dsyrk('L', 'N', n, n, 1.0, 1.0);
  }
}
