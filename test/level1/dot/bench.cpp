#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dot.h"

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
        exo_sdot(nullptr, n, exo_win_1f32c{x.data(), {1}}, exo_win_1f32c{y.data(), {1}}, &result);
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SDOT) -> Args({1});
BENCHMARK(BM_EXO_SDOT) -> Args({1});


BENCHMARK(BM_CBLAS_SDOT) -> Args({2});
BENCHMARK(BM_EXO_SDOT) -> Args({2});


BENCHMARK(BM_CBLAS_SDOT) -> Args({4});
BENCHMARK(BM_EXO_SDOT) -> Args({4});


BENCHMARK(BM_CBLAS_SDOT) -> Args({8});
BENCHMARK(BM_EXO_SDOT) -> Args({8});


BENCHMARK(BM_CBLAS_SDOT) -> Args({16});
BENCHMARK(BM_EXO_SDOT) -> Args({16});


BENCHMARK(BM_CBLAS_SDOT) -> Args({32});
BENCHMARK(BM_EXO_SDOT) -> Args({32});


BENCHMARK(BM_CBLAS_SDOT) -> Args({64});
BENCHMARK(BM_EXO_SDOT) -> Args({64});


BENCHMARK(BM_CBLAS_SDOT) -> Args({128});
BENCHMARK(BM_EXO_SDOT) -> Args({128});


BENCHMARK(BM_CBLAS_SDOT) -> Args({256});
BENCHMARK(BM_EXO_SDOT) -> Args({256});


BENCHMARK(BM_CBLAS_SDOT) -> Args({512});
BENCHMARK(BM_EXO_SDOT) -> Args({512});


BENCHMARK(BM_CBLAS_SDOT) -> Args({1024});
BENCHMARK(BM_EXO_SDOT) -> Args({1024});


BENCHMARK(BM_CBLAS_SDOT) -> Args({2048});
BENCHMARK(BM_EXO_SDOT) -> Args({2048});


BENCHMARK(BM_CBLAS_SDOT) -> Args({4096});
BENCHMARK(BM_EXO_SDOT) -> Args({4096});


BENCHMARK(BM_CBLAS_SDOT) -> Args({8192});
BENCHMARK(BM_EXO_SDOT) -> Args({8192});


BENCHMARK(BM_CBLAS_SDOT) -> Args({16384});
BENCHMARK(BM_EXO_SDOT) -> Args({16384});


BENCHMARK(BM_CBLAS_SDOT) -> Args({32768});
BENCHMARK(BM_EXO_SDOT) -> Args({32768});


BENCHMARK(BM_CBLAS_SDOT) -> Args({65536});
BENCHMARK(BM_EXO_SDOT) -> Args({65536});


BENCHMARK(BM_CBLAS_SDOT) -> Args({131072});
BENCHMARK(BM_EXO_SDOT) -> Args({131072});


BENCHMARK(BM_CBLAS_SDOT) -> Args({262144});
BENCHMARK(BM_EXO_SDOT) -> Args({262144});


BENCHMARK(BM_CBLAS_SDOT) -> Args({524288});
BENCHMARK(BM_EXO_SDOT) -> Args({524288});


BENCHMARK(BM_CBLAS_SDOT) -> Args({1048576});
BENCHMARK(BM_EXO_SDOT) -> Args({1048576});


BENCHMARK(BM_CBLAS_SDOT) -> Args({2097152});
BENCHMARK(BM_EXO_SDOT) -> Args({2097152});


BENCHMARK(BM_CBLAS_SDOT) -> Args({4194304});
BENCHMARK(BM_EXO_SDOT) -> Args({4194304});


BENCHMARK(BM_CBLAS_SDOT) -> Args({8388608});
BENCHMARK(BM_EXO_SDOT) -> Args({8388608});


BENCHMARK(BM_CBLAS_SDOT) -> Args({16777216});
BENCHMARK(BM_EXO_SDOT) -> Args({16777216});


BENCHMARK(BM_CBLAS_SDOT) -> Args({33554432});
BENCHMARK(BM_EXO_SDOT) -> Args({33554432});


BENCHMARK(BM_CBLAS_SDOT) -> Args({67108864});
BENCHMARK(BM_EXO_SDOT) -> Args({67108864});
