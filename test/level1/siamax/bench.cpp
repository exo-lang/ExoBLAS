#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_isamax.h"

static void BM_cblas_isamax(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = AlignedBuffer<float>(N, incX);

    for (auto _ : state) {
        cblas_isamax(N, X.data(), incX);
    }
}

static void BM_exo_isamax(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = AlignedBuffer<float>(N, incX);

    for (auto _ : state) {
        exo_isamax(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_isamax)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_isamax)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_isamax)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
// BENCHMARK(BM_exo_isamax)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });