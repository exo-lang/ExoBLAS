#include <benchmark/benchmark.h>
#include <cblas.h>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "exo_sgemm.h"
#include "generate_buffer.h"

static void BM_cblas_sgemm(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  auto a = AlignedBuffer2D<float>(m, k);
  auto b = AlignedBuffer2D<float>(k, n);
  auto c = AlignedBuffer2D<float>(m, n);

  float alpha = 1.0f;
  float beta = 1.0f;

  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                a.data(), k, b.data(), n, beta, c.data(), n);
  }
}

static void BM_exo_sgemm(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  auto a = AlignedBuffer2D<float>(m, k);
  auto b = AlignedBuffer2D<float>(k, n);
  auto c = AlignedBuffer2D<float>(m, n);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  for (auto _ : state) {
    exo_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              a.data(), k, b.data(), n, beta, c.data(), n);
  }
}

static void gemm_arguments_increasing(benchmark::internal::Benchmark *b) {
  size_t L1 = 192 * 1024;
  size_t L2 = 3 * 1024 * 1024 / 2;
  size_t L3 = 9 * 1024 * 1024;
  size_t DRAM = L3 * 20;
  size_t minMem = 0;
  size_t maxMem = DRAM;

  int m_multiple = 32;
  int k_multiple = 96;
  int n_multiple = 24;
  for (int i = 1; i < 1000; i += 4) {
    int M = i * m_multiple;
    int N = i * n_multiple;
    int K = i * k_multiple;

    size_t total = (size_t)M * N + (size_t)M * K + (size_t)N * K;
    total *= 4;

    if (total < maxMem && total >= minMem) {
      b->Args({M, N, K});
    }
  }
}

static void gemm_arguments_large(benchmark::internal::Benchmark *b) {
  for (int i = 1; i < 4; ++i) {
    b->Args({4096 * i, 3072 * i, 96 * 10 * i});
  }
}

BENCHMARK(BM_cblas_sgemm)
    ->ArgNames({"m", "n", "k"})
    ->Apply(gemm_arguments_large);

BENCHMARK(BM_exo_sgemm)->ArgNames({"m", "n", "k"})->Apply(gemm_arguments_large);

// BENCHMARK(BM_cblas_sgemm)
//     ->ArgNames({"m", "n", "k"})
//     ->Apply(gemm_arguments_increasing);

// BENCHMARK(BM_exo_sgemm)->ArgNames({"m", "n",
// "k"})->Apply(gemm_arguments_increasing);
