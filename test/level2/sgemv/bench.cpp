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

class GEMVFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State& state) {
    n = state.range(0);
    a = gen_matrix(n, n);
    x = gen_matrix(n, 1);
    y = gen_matrix(n, 1);
    alpha = 0.9f;
    beta = 0.7f;
  }

  void TearDown(const ::benchmark::State& state) {
    a.clear();
    x.clear();
    y.clear();
  }

  int n;
  std::vector<float> a;
  std::vector<float> x;
  std::vector<float> y;
  float alpha;
  float beta;
};

BENCHMARK_DEFINE_F(GEMVFixture, NAIVE)(benchmark::State& state) {
  for (auto _ : state) {
    naive_sgemv_square(&alpha, &beta, a.data(), x.data(), y.data(), n, n);
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 3 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );
}

BENCHMARK_DEFINE_F(GEMVFixture, CBLAS)(benchmark::State& state) {
  for (auto _ : state) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 3 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );
}

BENCHMARK_DEFINE_F(GEMVFixture, APPLE_GEMM)(benchmark::State& state) {
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, n, alpha, a.data(), n, x.data(), 1, beta, y.data(), 1);
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 3 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );
}

BENCHMARK_DEFINE_F(GEMVFixture, EXO_8)(benchmark::State& state) {
  for (auto _ : state) {
    sgemv_stride_1(nullptr, &alpha, &beta, n, n, a.data(), x.data(), y.data());
  }

  state.counters["flops"] = benchmark::Counter(
    static_cast<double>(state.iterations()) * 3 * n * n,
    benchmark::Counter::kIsRate,
    benchmark::Counter::kIs1000
  );

  auto y2 = y;
  auto y3 = y;
  naive_sgemv_square(&alpha, &beta, a.data(), x.data(), y2.data(), n, n);
  sgemv_stride_1(nullptr, &alpha, &beta, n, n, a.data(), x.data(), y3.data());

  for (int i = 0; i < y2.size(); i++) {
    float expected = y2[i];
    float actual = y3[i];
    double relerr = fabsf(actual - expected) / expected;
    if (relerr > 1e-2) {
      printf("index %d: %.6f != %.6f (expected)\n", i, actual, expected);
    }
  }
}

// Register the function as a benchmark
// BENCHMARK_REGISTER_F(GEMVFixture, NAIVE) -> Range(16, 16384);
BENCHMARK_REGISTER_F(GEMVFixture, CBLAS) -> ArgNames({"n"}) -> Range(16, 16384);
// BENCHMARK_REGISTER_F(GEMVFixture, APPLE_GEMM) -> Range(16, 16384);
BENCHMARK_REGISTER_F(GEMVFixture, EXO_8) -> ArgNames({"n"}) -> Range(16, 16384);