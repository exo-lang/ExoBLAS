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

#include "exo_sgemm.h"
#include "generate_buffer.h"

static void BM_SGEMM_CBLAS(benchmark::State &state) {
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

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * m * n * k,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

static void BM_SGEMM_EXO(benchmark::State &state) {
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

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * m * n * k,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_SGEMM_CBLAS)
    ->ArgNames({"n", "m", "k"})
    ->Args({1, 1, 1})
    ->Args({4, 4, 4})
    ->Args({48, 48, 48})
    ->Args({48 * 2, 48 * 2, 48 * 2})
    ->Args({48 * 4, 48 * 4, 48 * 4})
    ->Args({48 * 8, 48 * 8, 48 * 8})
    ->Args({528, 240, 528})
    ->Args({1056, 240, 528})
    ->Args({1056 * 2, 240, 528})
    ->Args({1056 * 2, 240 * 2, 528 * 2})
    ->Args({1056 * 2 * 2 * 2, 240 * 4 * 2, 528 * 2 * 2});
//    ->ArgsProduct({benchmark::CreateRange(48, 48*100, 48)});

BENCHMARK(BM_SGEMM_EXO)
    ->ArgNames({"n", "m", "k"})
    ->Args({1, 1, 1})
    ->Args({4, 4, 4})
    ->Args({48, 48, 48})
    ->Args({48 * 2, 48 * 2, 48 * 2})
    ->Args({48 * 4, 48 * 4, 48 * 4})
    ->Args({48 * 8, 48 * 8, 48 * 8})
    ->Args({528, 240, 528})
    ->Args({1056, 240, 528})
    ->Args({1056 * 2, 240, 528})
    ->Args({1056 * 2, 240 * 2, 528 * 2})
    ->Args({1056 * 2 * 2 * 2, 240 * 4 * 2, 528 * 2 * 2});
//->ArgsProduct({benchmark::CreateRange(48, 48*100, 48)});
