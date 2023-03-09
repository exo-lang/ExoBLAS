#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_snrm2.h"

static void BM_cblas_snrm2(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);

    auto X = AlignedBuffer<float>(N, incX);

    for (auto _ : state) {
        cblas_snrm2(N, X.data(), incX);
    }
}

static void BM_exo_snrm2(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);

    auto X = AlignedBuffer<float>(N, incX);

    for (auto _ : state) {
        exo_snrm2(N, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_snrm2)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });
BENCHMARK(BM_exo_snrm2)->ArgNames({"n", "incX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}
    });

// BENCHMARK(BM_cblas_snrm2)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
// BENCHMARK(BM_exo_snrm2)->ArgNames({"n", "incX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
