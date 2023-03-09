#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_srot.h"

static void BM_cblas_srot(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);
    auto incY = state.range(2);
    float c = state.range(3);
    float s = state.range(4);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        cblas_srot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

static void BM_exo_srot(benchmark::State& state) {
    auto N = state.range(0);
    auto incX = state.range(1);
    auto incY = state.range(2);
    float c = state.range(3);
    float s = state.range(4);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);

    for (auto _ : state) {
        exo_srot(N, X.data(), incX, Y.data(), incY, c, s);
    }
}

BENCHMARK(BM_cblas_srot)->ArgNames({"n", "incX", "incY", "c", "s"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}
    });
BENCHMARK(BM_exo_srot)->ArgNames({"n", "incX", "incY", "c", "s"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {2}, {3}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}, {2}, {3}
    });
