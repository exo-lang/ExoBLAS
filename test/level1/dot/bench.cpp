#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dot.h"

static float naive_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY) {
    float result = 0.0;
    for (int i = 0; i < N; ++i) {
        result += X[i * incX] + Y[i * incY];
    }
    return result;
}

static void BM_NAIVE_SDOT(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(naive_sdot(n, x.data(), 1, y.data(), 1));
    }

    // state.counters["flops"] = ;
}

static void BM_CBLAS_SDOT(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);

    for (auto _ : state) {
        cblas_sdot(n, x.data(), 1, y.data(), 1);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SDOT(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    float result = 0.0;

    for (auto _ : state) {
        exo_sdot(nullptr, n, exo_win_1f32c{x.data(), 1}, exo_win_1f32c{y.data(), 1}, &result);
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SDOT) -> Args({64});
BENCHMARK(BM_NAIVE_SDOT) -> Args({64});
BENCHMARK(BM_EXO_SDOT) -> Args({64});

BENCHMARK(BM_CBLAS_SDOT) -> Args({256});
BENCHMARK(BM_NAIVE_SDOT) -> Args({256});
BENCHMARK(BM_EXO_SDOT) -> Args({256});

BENCHMARK(BM_CBLAS_SDOT) -> Args({512});
BENCHMARK(BM_NAIVE_SDOT) -> Args({512});
BENCHMARK(BM_EXO_SDOT) -> Args({512});

BENCHMARK(BM_CBLAS_SDOT) -> Args({100000});
BENCHMARK(BM_NAIVE_SDOT) -> Args({100000});
BENCHMARK(BM_EXO_SDOT) -> Args({100000});

BENCHMARK(BM_CBLAS_SDOT) -> Args({1000000});
BENCHMARK(BM_NAIVE_SDOT) -> Args({1000000});
BENCHMARK(BM_EXO_SDOT) -> Args({1000000});

BENCHMARK(BM_CBLAS_SDOT) -> Args({10000000});
BENCHMARK(BM_NAIVE_SDOT) -> Args({10000000});
BENCHMARK(BM_EXO_SDOT) -> Args({10000000});
