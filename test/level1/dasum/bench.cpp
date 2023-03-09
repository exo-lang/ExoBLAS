#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dasum.h"

static void BM_cblas_dasum(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        cblas_dasum(N, X.data(), incX);
    }
}

static void BM_exo_dasum(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        exo_dasum(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_dasum)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_dasum)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_dasum)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
// BENCHMARK(BM_exo_dasum)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });