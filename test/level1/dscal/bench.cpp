#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dscal.h"

static void BM_cblas_dscal(benchmark::State& state) {
    auto N = state.range(0);
    auto alpha = state.range(1);
    auto incX = state.range(2);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        cblas_dscal(N, alpha, X.data(), incX);
    }
}

static void BM_exo_dscal(benchmark::State& state) {
    auto N = state.range(0);
    auto alpha = state.range(1);
    auto incX = state.range(2);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        exo_dscal(N, alpha, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_dscal)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {17}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {17}, {1}
    });
BENCHMARK(BM_exo_dscal)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {17}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {17}, {1}
    });

// BENCHMARK(BM_cblas_dscal)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });
// BENCHMARK(BM_exo_dscal)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });