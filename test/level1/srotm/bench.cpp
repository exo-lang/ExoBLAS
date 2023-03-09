#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_srotm.h"

static void BM_cblas_srotm(benchmark::State& state) {
    int N = state.range(0);
    float HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);
    auto H = generate1d_sbuffer(5, 1);
    H[0] = HFlag;
    H[1] = 1.2;
    H[2] = 2.2;
    H[3] = 3.2;
    H[4] = 4.2;

    for (auto _ : state) {
        cblas_srotm(N, X.data(), incX, Y.data(), incY, H.data());
    }
}

static void BM_EXO_SROTM(benchmark::State& state) {
    int N = state.range(0);
    float HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);
    auto H = generate1d_sbuffer(5, 1);
    H[0] = HFlag;
    H[1] = 1.2;
    H[2] = 2.2;
    H[3] = 3.2;
    H[4] = 4.2;

    for (auto _ : state) {
        exo_srotm(N, X.data(), incX, Y.data(), incY, H.data());
    }
}

// Register the function as a benchmark
BENCHMARK(BM_cblas_srotm)->ArgNames({"n", "HFlag", "incX", "incY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}
    });
BENCHMARK(BM_EXO_SROTM)->ArgNames({"n", "HFlag", "incX", "incY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}
    });