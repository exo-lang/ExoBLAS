#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_sdsdot_h.h"

static void BM_cblas_sdsdot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    float alpha = state.range(3);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        cblas_sdsdot(N, alpha, X.data(), incX, Y.data(), incY);
    }
}

static void BM_exo_sdsdot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    float alpha = state.range(3);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        exo_sdsdot(N, alpha, X.data(), incX, Y.data(), incY);
    }
}

BENCHMARK(BM_cblas_sdsdot)->ArgNames({"n", "incX", "incY", "alpha"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {1}
    });
BENCHMARK(BM_exo_sdsdot)->ArgNames({"n", "incX", "incY", "alpha"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {1}
    });

// BENCHMARK(BM_cblas_sdsdot)->ArgNames({"n", "incX", "incY", "alpha"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {1}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {1}
//     });
// BENCHMARK(BM_exo_sdsdot)->ArgNames({"n", "incX", "incY", "alpha"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {1}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {1}
//     });
