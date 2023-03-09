#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_saxpy.h"

static void BM_cblas_saxpy(benchmark::State& state) {
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);

    for (auto _ : state) {
        cblas_saxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }
}

static void BM_exo_saxpy(benchmark::State& state) {
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);

    for (auto _ : state) {
        exo_saxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }
}

BENCHMARK(BM_cblas_saxpy)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}, {64}, {64}
    });
BENCHMARK(BM_exo_saxpy)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}, {64}, {64}
    });

// BENCHMARK(BM_cblas_saxpy)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {3}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {3}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
// BENCHMARK(BM_exo_saxpy)->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {3}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {3}, {-4, 2, 1, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
