#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <chrono>

#include "exo_gemv.h"
#include <cblas.h>
#include <benchmark/benchmark.h>

void naive_sgemv_square(const float* alpha, const float* beta, const float *a, const float *x, float *y, long m, long n) {
  for (long i = 0; i < m; i++) {
    y[i] = *beta * y[i];
    for (long j = 0; j < n; j++) {
      y[i] += *alpha * a[i * n + j] * x[j];
    }
  }
}

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

static void BM_GEMV_NAIVE(benchmark::State& state) {
  int n = state.range(0);

  auto a = gen_matrix(n, n);
  auto x = gen_matrix(n, 1);
  auto y = gen_matrix(n, 1);

  float alpha = 0.9f;
  float beta = 0.7f;

  for (auto _ : state) {
    naive_sgemv_square(&alpha, &beta, a.data(), x.data(), y.data(), n, n);
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 2 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );
}


static void BM_GEMV_APPLE(benchmark::State& state) {
  int n = state.range(0);

  auto a = gen_matrix(n, n);
  auto x = gen_matrix(n, 1);
  auto y = gen_matrix(n, 1);

  float alpha = 0.9f;
  float beta = 0.7f;

  for (auto _ : state) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 2 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );
}


static void BM_GEMV_EXO(benchmark::State& state) {
  int n = state.range(0);

  auto a = gen_matrix(n, n);
  auto x = gen_matrix(n, 1);
  auto y = gen_matrix(n, 1);

  float alpha = 0.9f;
  float beta = 0.7f;

  for (auto _ : state) {
    sgemv_exo_v2(nullptr, &alpha, &beta, n, n, n, a.data(), x.data(), y.data());
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 2 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );

  // write correctness test here
}

// Register the function as a benchmark
BENCHMARK(BM_GEMV_NAIVE) -> Args({3000});
BENCHMARK(BM_GEMV_APPLE) -> Args({3000});
BENCHMARK(BM_GEMV_EXO) -> Args({3000});

BENCHMARK(BM_GEMV_NAIVE) -> Args({1000});
BENCHMARK(BM_GEMV_APPLE) -> Args({1000});
BENCHMARK(BM_GEMV_EXO) -> Args({1000});

BENCHMARK(BM_GEMV_NAIVE) -> Args({100});
BENCHMARK(BM_GEMV_APPLE) -> Args({100});
BENCHMARK(BM_GEMV_EXO) -> Args({100});