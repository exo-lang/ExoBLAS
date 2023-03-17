#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_sger.h"

static void BM_cblas_sger(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    int lda = N;
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);
    size_t alignmentA = state.range(6);

    auto X = AlignedBuffer<float>(M, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);
    auto A = AlignedBuffer<float>(M * lda, 1, alignmentA);

    for (auto _ : state) {
        cblas_sger(CBLAS_ORDER::CblasRowMajor, M, N, alpha, X.data(), incX, Y.data(), incY, A.data(), lda);
    }
}

static void BM_exo_sger(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    int lda = N;
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);
    size_t alignmentA = state.range(6);

    auto X = AlignedBuffer<float>(M, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);
    auto A = AlignedBuffer<float>(M * lda, 1, alignmentA);

    for (auto _ : state) {
        exo_sger(M, N, alpha, X.data(), incX, Y.data(), incY, A.data(), lda);
    }
}

BENCHMARK(BM_cblas_sger)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY", "alignmentA"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 13), 2), {0, 1, 3}, {1}, {1}, {64}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 13) - 1, 7), {0, 1, 3}, {1}, {1}, {64}, {64}, {64}
    });
BENCHMARK(BM_exo_sger)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY", "alignmentA"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 13), 2), {0, 1, 3}, {1}, {1}, {64}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 13) - 1, 7), {0, 1, 3}, {1}, {1}, {64}, {64}, {64}
    });
