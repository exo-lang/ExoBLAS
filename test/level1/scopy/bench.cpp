#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_scopy_wrapper.h"

static void BM_CBLAS_SCOPY(benchmark::State& state) {
    int n = state.range(0);
    int incx = state.range(1);
    int incy = state.range(2);

    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);

    for (auto _ : state) {
        cblas_scopy(n, x.data(), incx, y.data(), incy);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SCOPY(benchmark::State& state) {
    int n = state.range(0);
    int incx = state.range(1);
    int incy = state.range(2);

    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);

    for (auto _ : state) {
        exo_scopy(n, x.data(), incx, y.data(), incy);
    }

    // state.counters["flops"] = ;
}

// Run scopy with stride = 1
BENCHMARK(BM_CBLAS_SCOPY)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}
    });
BENCHMARK(BM_EXO_SCOPY)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}
    });

// Run scopy with stride != 1
// BENCHMARK(BM_CBLAS_SCOPY)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
// BENCHMARK(BM_EXO_SCOPY)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
