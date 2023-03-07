#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_sasum.h"

static void BM_cblas_sasum(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = generate1d_sbuffer(N, incX);

    for (auto _ : state) {
        cblas_sasum(N, X.data(), incX);
    }
}

static void BM_exo_sasum(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = generate1d_sbuffer(N, incX);

    for (auto _ : state) {
        exo_sasum(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_sasum)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_sasum)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_sasum)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });
// BENCHMARK(BM_exo_sasum)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });