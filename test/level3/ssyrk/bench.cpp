#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_syrk.h"
#include <cblas.h>
#include <benchmark/benchmark.h>

#define EPSILON 0.01

bool AreSame(double a, double b)
{
    return fabs(a - b) < EPSILON;
}


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


void test_syrk_correctness(benchmark::State& state) {

    int n = state.range(0);
    auto a = gen_matrix(n, n, -1.0);
    auto c = gen_matrix(n, n, 2.0);
    auto c2 = gen_matrix(n, n, 2.0);

    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, n, // M N
                1.0, // alpha
                a.data(),
                n, // M
                1.0,
                c.data(),
                n  // M
                );

    syrk_lower_notranspose(nullptr, n, n, a.data(), transpose(a, n, n).data(), c2.data());

    for (int i=0; i<n*n; i++) {
        double correct = std::round(c[i] * 1000.0) / 1000.0;
        double exo_out = std::round(c2[i] * 1000.0) / 1000.0;
        if (!AreSame(correct, exo_out))
            std::cout<<"Error at "<< i/n <<", "<<i%n<< ". Expected: "<<correct<<", got: "<<exo_out<<std::endl;
        assert(AreSame(correct, exo_out));
    }
}


static void BM_SYRK_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n, 2.0);
    auto c = gen_matrix(n, n, 2.0);

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


static void BM_SYRK_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n, 2.0);
    auto c = gen_matrix(n, n, 2.0);

    float alpha = 1.0f;
    float beta = 1.0f;

    for (auto _: state) {
        syrk_lower_notranspose(nullptr, n, n, a.data(), a.data(), c.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

    test_syrk_correctness(state);

}

BENCHMARK(BM_SYRK_CBLAS) -> Args({64});
BENCHMARK(BM_SYRK_EXO) -> Args({64});

BENCHMARK(BM_SYRK_CBLAS) -> Args({128});
BENCHMARK(BM_SYRK_EXO) -> Args({128});

BENCHMARK(BM_SYRK_CBLAS) -> Args({256});
BENCHMARK(BM_SYRK_EXO) -> Args({256});

BENCHMARK(BM_SYRK_CBLAS) -> Args({256});
BENCHMARK(BM_SYRK_EXO) -> Args({256});

BENCHMARK(BM_SYRK_CBLAS) -> Args({1024});
BENCHMARK(BM_SYRK_EXO) -> Args({1024});

BENCHMARK(BM_SYRK_CBLAS) -> Args({2048});
BENCHMARK(BM_SYRK_EXO) -> Args({2048});

BENCHMARK(BM_SYRK_CBLAS) -> Args({4096});
BENCHMARK(BM_SYRK_EXO) -> Args({4096});
