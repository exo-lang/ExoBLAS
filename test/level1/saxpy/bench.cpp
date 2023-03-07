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

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        cblas_saxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }
}

static void BM_exo_saxpy(benchmark::State& state) {
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        exo_saxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }
}

BENCHMARK(BM_cblas_saxpy)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}
    });
BENCHMARK(BM_exo_saxpy)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}
    });

// BENCHMARK(BM_cblas_saxpy)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
// BENCHMARK(BM_exo_saxpy)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
