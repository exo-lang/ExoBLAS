#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>


#include "exo_gemm.h"
#include <cblas.h>
#include <benchmark/benchmark.h>

#define EPSILON 0.01

bool AreSame(double a, double b)
{
    return fabs(a - b) < EPSILON;
}


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

static std::vector<float> transpose(std::vector<float> V, const int m, const int k ) {
    std::vector<float> V_t(k*m);
    for (int i=0; i<m; i++) {
        for (int j=0; j<k; j++) {
            V_t[j*m + i] = V[i*k + j];
        }
    }

    return V_t;
}


static void test_sgemm_correctness(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n);
    auto b = gen_matrix(n, n);
    auto c = gen_matrix(n, n);
    auto c2 = c; 
    const float alpha = 1.0;
    const float beta = 1.0;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  n, n, n, 
                  alpha, 
                  a.data(), n,
                  b.data(), n,
                  beta,
                  c.data(), n);

    exo_gemm(nullptr, n, n, n, &alpha, &beta, c2.data(), a.data(), b.data());

    for (int i=0; i<n*n; i++) {
        double correct = std::round(c[i] * 1000.0) / 1000.0;
        double exo_out = std::round(c2[i] * 1000.0) / 1000.0;
        if (!AreSame(correct, exo_out))
            std::cout<<"Error at "<< i/n <<", "<<i%n<< ". Expected: "<<correct<<", got: "<<exo_out<<std::endl;
        assert(AreSame(correct, exo_out));
    }
}


static void BM_GEMM_CBLAS(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n);
    auto b = gen_matrix(n, n);
    auto c = gen_matrix(n, n);

    float alpha = 1.0f;
    float beta = 1.0f;

    for (auto _: state) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0,
                a.data(), n,
                b.data(), n,
                1.0,
                c.data(), n);
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

}


static void BM_GEMM_EXO(benchmark::State& state) {
    int n = state.range(0);
    auto a = gen_matrix(n, n);
    auto b = gen_matrix(n, n);
    auto c = gen_matrix(n, n);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    for (auto _: state) {
        exo_gemm(nullptr, n, n, n, &alpha, &beta, c.data(), a.data(), b.data());
    }

    state.counters["flops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * n * n * n,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000	
    );

    test_sgemm_correctness(state);

}

BENCHMARK(BM_GEMM_CBLAS)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });
BENCHMARK(BM_GEMM_EXO)->ArgNames({"n"})->ArgsProduct({
      benchmark::CreateRange(32, (32 << 8), 2),
    })->ArgsProduct({
      benchmark::CreateRange(33, (32 << 8) - 1, 3),
    });


