#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dscal.h"

static void BM_cblas_dscal(benchmark::State& state) {
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    size_t alignmentX = state.range(3);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);

    for (auto _ : state) {
        cblas_dscal(N, alpha, X.data(), incX);
    }
}

static void BM_exo_dscal(benchmark::State& state) {
    int N = state.range(0);
    float alpha = state.range(1);
    int incX = state.range(2);
    size_t alignmentX = state.range(3);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);

    for (auto _ : state) {
        exo_dscal(N, alpha, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_dscal)->ArgNames({"n", "alpha", "incX", "alignmentX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {17}, {1}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {17}, {1}, {64}
    });
BENCHMARK(BM_exo_dscal)->ArgNames({"n", "alpha", "incX", "alignmentX"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {17}, {1}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {17}, {1}, {64}
    });

// BENCHMARK(BM_cblas_dscal)->ArgNames({"n", "alpha", "incX", "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {17}, {-4, 2, 4, 10}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {17}, {-4, 2, 4, 10}, {64}
//     });
// BENCHMARK(BM_exo_dscal)->ArgNames({"n", "alpha", "incX", "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {17}, {-4, 2, 4, 10}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {17}, {-4, 2, 4, 10}, {64}
//     });