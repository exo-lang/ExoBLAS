#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dnrm2.h"

static void BM_cblas_dnrm2(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        cblas_dnrm2(N, X.data(), incX);
    }
}

static void BM_exo_dnrm2(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);

    auto X = generate1d_dbuffer(N, incX);

    for (auto _ : state) {
        exo_dnrm2(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_dnrm2)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_dnrm2)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_dnrm2)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });
// BENCHMARK(BM_exo_dnrm2)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}
//     });
