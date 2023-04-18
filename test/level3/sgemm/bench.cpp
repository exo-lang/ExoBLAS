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

static void print_matrix(std::vector<float> M, int n, int k) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << M[j * k + i] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

static void BM_SGEMM_CBLAS(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n);
  auto b = AlignedBuffer2D<float>(n, n);
  auto c = AlignedBuffer2D<float>(n, n);

  float alpha = 1.0f;
  float beta = 1.0f;

  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha,
                a.data(), n, b.data(), n, beta, c.data(), n);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * n * n * n,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

static void BM_SGEMM_EXO(benchmark::State &state) {
  int n = state.range(0);
  auto a = AlignedBuffer2D<float>(n, n);
  auto b = AlignedBuffer2D<float>(n, n);
  auto c = AlignedBuffer2D<float>(n, n);

  const float alpha = 1.0f;
  const float beta = 1.0f;

  for (auto _ : state) {
    exo_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, &alpha, &beta,
              a.data(), b.data(), c.data());
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * 2 * n * n * n,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(BM_SGEMM_CBLAS)
    ->ArgNames({"n"})
    ->Arg(48)->Arg(48*2)->Arg(48*4)->Arg(48*8)->Arg(48*16)->Arg(48*32)->Arg(48*64)->Arg(48*128)->Arg(48*128*2);
//    ->ArgsProduct({benchmark::CreateRange(48, 48*100, 48)});

BENCHMARK(BM_SGEMM_EXO)->ArgNames({"n"})
    ->ArgNames({"n"})
    ->Arg(48)->Arg(48*2)->Arg(48*4)->Arg(48*8)->Arg(48*16)->Arg(48*32)->Arg(48*64)->Arg(48*128)->Arg(48*128*2);
    //->ArgsProduct({benchmark::CreateRange(48, 48*100, 48)});


