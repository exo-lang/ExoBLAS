#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_idamax.h"

static void BM_cblas_idamax(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = AlignedBuffer<double>(N, incX);

    for (auto _ : state) {
        cblas_idamax(N, X.data(), incX);
    }
}

static void BM_exo_idamax(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);

    auto X = AlignedBuffer<double>(N, incX);

    for (auto _ : state) {
        exo_idamax(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_idamax)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_idamax)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_idamax)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
// BENCHMARK(BM_exo_idamax)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });