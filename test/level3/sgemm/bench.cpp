#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_sgemm.h"
#include "generate_buffer.h"
#include <cblas.h>
#include <benchmark/benchmark.h>


static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

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


static void BM_SGEMM_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<float>(n, n);
    auto b = AlignedBuffer2D<float>(n, n);
    auto c = AlignedBuffer2D<float>(n, n);

    float alpha = 1.0f;
    float beta = 1.0f;

    for (auto _: state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                a.data(), n,
                b.data(), n,
                beta,
                c.data(), n);
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}


static void BM_SGEMM_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n);
    auto b = gen_matrix(n, n);
    auto c = gen_matrix(n, n);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    for (auto _: state) {
        exo_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &alpha, &beta, a.data(), b.data(), c.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}

BENCHMARK(BM_SGEMM_CBLAS)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });
BENCHMARK(BM_SGEMM_EXO)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });


