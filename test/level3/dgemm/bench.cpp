#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_dgemm.h"
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


static void BM_DGEMM_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<double>(n, n);
    auto b = AlignedBuffer2D<double>(n, n);
    auto c = AlignedBuffer2D<double>(n, n);

    double alpha = 1.0f;
    double beta = 1.0f;

    for (auto _: state) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                alpha,
                a.data(), n,
                b.data(), n,
                beta,
                c.data(), n);
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n * 2,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}


static void BM_DGEMM_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<double>(n, n);
    auto b = AlignedBuffer2D<double>(n, n);
    auto c = AlignedBuffer2D<double>(n, n);

    const double alpha = 1.0f;
    const double beta = 1.0f;

    for (auto _: state) {
        exo_dgemm('N', n, n, n, &alpha, &beta, a.data(), b.data(), c.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n * 2,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({4096});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({4096});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({2048});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({2048});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({1024});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({1024});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({512});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({512});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({256});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({256});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({128});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({128});

BENCHMARK(BM_DGEMM_CBLAS) -> ArgNames({"n"}) -> Args({64});
BENCHMARK(BM_DGEMM_EXO) -> ArgNames({"n"}) -> Args({64});
