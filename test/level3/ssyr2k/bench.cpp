#include <benchmark/benchmark.h>
#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "exo_ssyr2k.h"
#include "generate_buffer.h"

static void print_matrix(std::vector<float> M, int n, int k) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << M[j * k + i] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

static std::vector<float> transpose(std::vector<float> V, const int m,
                                    const int k) {
  std::vector<float> V_t(k * m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      V_t[j * m + i] = V[i * k + j];
    }
  }

  return V_t;
}

static void BM_SSYR2K_CBLAS(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n, 2.0, 64);
  auto b = a;
  auto c = AlignedBuffer2D<float>(n, n, 2.0, 64);

  for (auto _ : state) {
    cblas_ssyr2k(CblasRowMajor, CblasLower, CblasNoTrans, n, n,  // M N
                 1.0,                                            // alpha
                 a.data(),
                 n,  // M
                 b.data(), n, 1.0, c.data(),
                 n  // M
    );
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * n * n * n * 2,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

static void BM_SSYR2K_EXO(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n, 2.0, 64);
  auto c = AlignedBuffer2D<float>(n, n, 2.0, 64);

  float alpha = 1.0f;
  float beta = 1.0f;

  for (auto _ : state) {
    exo_ssyr2k(CblasRowMajor, CblasLower, CblasNoTrans, n, n, &alpha, a.data(),
               n, a.data(), n, &beta, c.data(), n);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * n * n * n * 2,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_SSYR2K_CBLAS)
    ->ArgNames({"n"})
    ->RangeMultiplier(2)
    ->Range(16, 8192);
BENCHMARK(BM_SSYR2K_EXO)->ArgNames({"n"})->RangeMultiplier(2)->Range(16, 8192);
