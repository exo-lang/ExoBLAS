#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dcopy_wrapper.h"

static void BM_CBLAS_DCOPY(benchmark::State& state) {
    int n = state.range(0);
    int incx = state.range(1);
    int incy = state.range(2);

    auto x = generate1d_dbuffer(n, incx);
    auto y = generate1d_dbuffer(n, incy);

    for (auto _ : state) {
        cblas_dcopy(n, x.data(), incx, y.data(), incy);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_DCOPY(benchmark::State& state) {
    int n = state.range(0);
    int incx = state.range(1);
    int incy = state.range(2);

    auto x = generate1d_dbuffer(n, incx);
    auto y = generate1d_dbuffer(n, incy);

    for (auto _ : state) {
        exo_dcopy(n, x.data(), incx, y.data(), incy);
    }

    // state.counters["flops"] = ;
}

// Run dcopy with stride = 1
BENCHMARK(BM_CBLAS_DCOPY)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}
    });
BENCHMARK(BM_EXO_DCOPY)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {1}
    });

// Run dcopy with stride != 1
// BENCHMARK(BM_CBLAS_DCOPY)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
// BENCHMARK(BM_EXO_DCOPY)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
