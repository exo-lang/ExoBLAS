#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_drot.h"

static void BM_cblas_drot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    double c = state.range(3);
    double s = state.range(4);

    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);

    for (auto _ : state) {
        cblas_drot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

static void BM_exo_drot(benchmark::State& state) {
    int N = state.range(0);
    int incX = state.range(1);
    int incY = state.range(2);
    double c = state.range(3);
    double s = state.range(4);

    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);

    for (auto _ : state) {
        exo_drot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

BENCHMARK(BM_cblas_drot)->ArgNames({"n", "incX", "incY", "c", "s"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}
    });
BENCHMARK(BM_exo_drot)->ArgNames({"n", "incX", "incY", "c", "s"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}
    });
