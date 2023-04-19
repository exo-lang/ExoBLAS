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

#include "exo_ssymm.h"
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

static void BM_SSYMM_CBLAS(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n);
  auto b = AlignedBuffer2D<float>(n, n);
  auto c = AlignedBuffer2D<float>(n, n);

  float alpha = 1.0f;
  float beta = 1.0f;

  for (auto _ : state) {
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower, n, n, alpha, a.data(), n,
                b.data(), n, beta, c.data(), n);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * n * n * n,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

static void BM_SSYMM_EXO(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n);
  auto b = AlignedBuffer2D<float>(n, n);
  auto c = AlignedBuffer2D<float>(n, n);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  for (auto _ : state) {
    exo_ssymm(CblasRowMajor, CblasLeft, CblasLower, n, n, &alpha, a.data(), n,
              b.data(), n, &beta, c.data(), n);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * n * n * n,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_SSYMM_CBLAS)
    ->ArgNames({"n", "m", "k"})
    ->ArgNames({"n", "m", "k"})
    ->Args({480, 480, 480})
    ->Args({4800, 4800, 4800});
BENCHMARK(BM_SSYMM_EXO)
    ->ArgNames({"n", "m", "k"})
    ->Args({480, 480, 480})
    ->Args({4800, 4800, 4800});
