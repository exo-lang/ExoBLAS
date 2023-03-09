#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_drot.h"

static void BM_cblas_drot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    double c = state.range(3);
    double s = state.range(4);
    size_t alignmentX = state.range(5);
    size_t alignmentY = state.range(6);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        cblas_drot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

static void BM_exo_drot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    double c = state.range(3);
    double s = state.range(4);
    size_t alignmentX = state.range(5);
    size_t alignmentY = state.range(6);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        exo_drot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

BENCHMARK(BM_cblas_drot)->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}, {64}, {64}
    });
BENCHMARK(BM_exo_drot)->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}, {64}, {64}
    });

// BENCHMARK(BM_cblas_drot)->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {2}, {3}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {2}, {3}, {64}, {64}
//     });
// BENCHMARK(BM_exo_drot)->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {2}, {3}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {2}, {3}, {64}, {64}
//     });