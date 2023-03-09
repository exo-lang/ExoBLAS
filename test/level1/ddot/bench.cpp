#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_ddot.h"

static void BM_cblas_ddot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    size_t alignmentX = state.range(3);
    size_t alignmentY = state.range(4);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        cblas_ddot(N, X.data(), incX, Y.data(), incY);
    }
}

static void BM_exo_ddot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    size_t alignmentX = state.range(3);
    size_t alignmentY = state.range(4);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        exo_ddot(N, X.data(), incX, Y.data(), incY);
    }
}

BENCHMARK(BM_cblas_ddot)->ArgNames({"n", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {64}, {64}
    });
BENCHMARK(BM_exo_ddot)->ArgNames({"n", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {64}, {64}
    });

// BENCHMARK(BM_cblas_ddot)->ArgNames({"n", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
// BENCHMARK(BM_exo_ddot)->ArgNames({"n", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
