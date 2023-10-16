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
  int n = state.range(0);
  int m = state.range(1);
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
  int n = state.range(0);
  int m = state.range(1);
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

static void gemm_arguments(benchmark::internal::Benchmark *b) {
  int L1 = 192 * 1024 / 4;
  int max_N = sqrt(L1 / 3);

  int multiple = 24;
  for (int i = 1; i < 1000; ++i) {
    int N = i * multiple;
    if (N < max_N) {
      b->Args({N, N, N});
    } else {
      break;
    }
  }
}

BENCHMARK(BM_cblas_sgemm)->ArgNames({"n", "m", "k"})->Apply(gemm_arguments);

BENCHMARK(BM_exo_sgemm)->ArgNames({"n", "m", "k"})->Apply(gemm_arguments);
