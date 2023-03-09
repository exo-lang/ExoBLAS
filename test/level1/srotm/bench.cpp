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
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);
    float H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

    for (auto _ : state) {
        cblas_srotm(N, X.data(), incX, Y.data(), incY, H);
    }
}

static void BM_exo_srotm(benchmark::State& state) {
    int N = state.range(0);
    float HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto Y = AlignedBuffer<float>(N, incY, alignmentY);
    float H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

    for (auto _ : state) {
        exo_srotm(N, X.data(), incX, Y.data(), incY, H);
    }
}

BENCHMARK(BM_cblas_srotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    });
BENCHMARK(BM_exo_srotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    });

// BENCHMARK(BM_cblas_srotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2),
//       {-1, 0, 1, -2}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//     benchmark::CreateRange(7, (1 << 26) - 1, 7),
//     {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
//     });
// BENCHMARK(BM_exo_srotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2),
//       {-1, 0, 1, -2}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7),
//       {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
//     });