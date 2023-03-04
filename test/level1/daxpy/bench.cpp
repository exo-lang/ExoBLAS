#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_daxpy_wrapper.h"

static void BM_cblas_daxpy(benchmark::State& state) {
    int N = state.range(0);
    double alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    std::vector<double> X = generate1d_dbuffer(N, incX);
    std::vector<double> Y = generate1d_dbuffer(N, incY);

    for (auto _ : state) {
        cblas_daxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }

    // state.counters["flops"] = ;
}

static void BM_exo_daxpy(benchmark::State& state) {
    int N = state.range(0);
    double alpha = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);

    std::vector<double> X = generate1d_dbuffer(N, incX);
    std::vector<double> Y = generate1d_dbuffer(N, incY);
    

    for (auto _ : state) {
        exo_daxpy(N, alpha, X.data(), incY, Y.data(), incY);
    }

    // state.counters["flops"] = ;
}

// Run daxpy with stride = 1
BENCHMARK(BM_cblas_daxpy)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}
    });
BENCHMARK(BM_exo_daxpy)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2), {3}, {1}, {1}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7), {3}, {1}, {1}
    });

// Run daxpy with stride != 1
// BENCHMARK(BM_cblas_daxpy)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
// BENCHMARK(BM_exo_daxpy)->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-10, -2, 1, 3, 7}, {-7, -1, 2, 4, 11}
//     });
