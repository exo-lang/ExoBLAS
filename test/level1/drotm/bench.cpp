#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_drotm.h"

static void BM_cblas_drotm(benchmark::State& state) {
    int N = state.range(0);
    double HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);
    auto H = generate1d_dbuffer(5, 1);
    H[0] = HFlag;
    H[1] = 1.2;
    H[2] = 2.2;
    H[3] = 3.2;
    H[4] = 4.2;

    for (auto _ : state) {
        cblas_drotm(N, X.data(), incX, Y.data(), incY, H.data());
    }
}

static void BM_EXO_SROTM(benchmark::State& state) {
    int N = state.range(0);
    double HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);
    auto H = generate1d_dbuffer(5, 1);
    H[0] = HFlag;
    H[1] = 1.2;
    H[2] = 2.2;
    H[3] = 3.2;
    H[4] = 4.2;

    for (auto _ : state) {
        exo_drotm(N, X.data(), incX, Y.data(), incY, H.data());
    }
}

// Register the function as a benchmark
BENCHMARK(BM_cblas_drotm)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}
    });
BENCHMARK(BM_EXO_SROTM)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}
    });