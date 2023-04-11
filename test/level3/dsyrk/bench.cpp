#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_dsyrk.h"
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


static void BM_DSYRK_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<double>(n, n, 2.0, 64);
    auto c = AlignedBuffer2D<double>(n, n, 2.0, 64);

    for (auto _: state) {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
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
        static_cast<double>(state.iterations()) * n * n * n * 2,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}


static void BM_DSYRK_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = AlignedBuffer2D<double>(n, n, 2.0, 64);
    auto c = AlignedBuffer2D<double>(n, n, 2.0, 64);

    double alpha = 1.0f;
    double beta = 1.0f;

    for (auto _: state) {
        exo_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, n, &alpha, a.data(), a.data(), &beta, c.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * n * n * n * 2,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );


}

BENCHMARK(BM_DSYRK_CBLAS)->ArgNames({"n"}) -> RangeMultiplier(2) -> Range(16, 8192);
BENCHMARK(BM_DSYRK_EXO)->ArgNames({"n"}) -> RangeMultiplier(2) -> Range(16, 8192);


