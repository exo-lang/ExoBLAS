#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_ssyrk.h"
#include "generate_buffer.h"
#include <cblas.h>
#include <benchmark/benchmark.h>


static std::vector<float> gen_matrix(long m, long n, float v) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  if (v==-1)
    std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });
  else    
    std::generate(std::begin(mat), std::end(mat), [&]() { return v; }); //Used for generating symmetric matrices

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

static std::vector<float> transpose(std::vector<float> V, const int m, const int k ) {
    std::vector<float> V_t(k*m);
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            V_t[j*m + i] = V[i*k + j];
        }
    }

    return V_t;
}


static void BM_SSYRK_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<float>(n, n, 2.0, 64);
    auto c = AlignedBuffer2D<float>(n, n, 2.0, 64);

    for (auto _: state) {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, n, // M N
                1.0, // alpha
                a.data(),
                n, // M
                1.0,
                c.data(),
                n  // M
                );
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}


static void BM_SSYRK_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<float>(n, n, 2.0, 64);
    auto c = AlignedBuffer2D<float>(n, n, 2.0, 64);

    float alpha = 1.0f;
    float beta = 1.0f;

    for (auto _: state) {
        exo_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, n, &alpha, a.data(), a.data(), &beta, c.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );


}

BENCHMARK(BM_SSYRK_CBLAS)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });
BENCHMARK(BM_SSYRK_EXO)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });
