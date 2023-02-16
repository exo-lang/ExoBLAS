#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_copy.h"

static void naive_scopy(const int N, const float *X, const int incX, float *Y, const int incY) {
    for (int i = 0; i < N; ++i) {
        Y[i * incY] = X[i * incX];
    }
}

static void BM_NAIVE_SCOPY(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);

    for (auto _ : state) {
        naive_scopy(n, x.data(), 1, y.data(), 1);
    }

    // state.counters["flops"] = ;
}

static void BM_CBLAS_SCOPY(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);

    for (auto _ : state) {
        cblas_scopy(n, x.data(), 1, y.data(), 1);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SCOPY(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);

    for (auto _ : state) {
        exo_scopy(nullptr, n, exo_win_1f32c{x.data(), 1}, exo_win_1f32{y.data(), 1});
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SCOPY) -> Args({64});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({64});
BENCHMARK(BM_EXO_SCOPY) -> Args({64});

BENCHMARK(BM_CBLAS_SCOPY) -> Args({256});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({256});
BENCHMARK(BM_EXO_SCOPY) -> Args({256});

BENCHMARK(BM_CBLAS_SCOPY) -> Args({512});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({512});
BENCHMARK(BM_EXO_SCOPY) -> Args({512});

BENCHMARK(BM_CBLAS_SCOPY) -> Args({100000});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({100000});
BENCHMARK(BM_EXO_SCOPY) -> Args({100000});

BENCHMARK(BM_CBLAS_SCOPY) -> Args({1000000});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({1000000});
BENCHMARK(BM_EXO_SCOPY) -> Args({1000000});

BENCHMARK(BM_CBLAS_SCOPY) -> Args({10000000});
BENCHMARK(BM_NAIVE_SCOPY) -> Args({10000000});
BENCHMARK(BM_EXO_SCOPY) -> Args({10000000});
