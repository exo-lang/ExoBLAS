#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_axpy.h"

static void BM_CBLAS_SAXPY(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    float alpha = 2.5;

    for (auto _ : state) {
        cblas_saxpy(n, alpha, x.data(), 1, y.data(), 1);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SAXPY(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    float alpha = 2.5;

    for (auto _ : state) {
        exo_saxpy(nullptr, n, &alpha, 
        exo_win_1f32c{.data = x.data(), .strides = {1}},
        exo_win_1f32{.data = y.data(), .strides = {1}});
    }

    x = generate1d_sbuffer(n, 1);
    y = generate1d_sbuffer(n, 1);
    std::vector<float> expected = generate1d_sbuffer(n, 1);

    for (int i = 0; i < n; ++i) {
        expected[i] = y[i] + alpha * x[i];
    }

    exo_saxpy(nullptr, n, &alpha, 
        exo_win_1f32c{.data = x.data(), .strides = {1}},
        exo_win_1f32{.data = y.data(), .strides = {1}});

    for (int i = 0; i < n; ++i) {
        if (y[i] != expected[i]) {
            printf("Failed ! got: %f, expected: %f, at index: %d\n", y[i], expected[i], i);
            exit(1);
        }
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SAXPY) -> Args({1});
BENCHMARK(BM_EXO_SAXPY) -> Args({1});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({2});
BENCHMARK(BM_EXO_SAXPY) -> Args({2});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({4});
BENCHMARK(BM_EXO_SAXPY) -> Args({4});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({8});
BENCHMARK(BM_EXO_SAXPY) -> Args({8});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({16});
BENCHMARK(BM_EXO_SAXPY) -> Args({16});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({32});
BENCHMARK(BM_EXO_SAXPY) -> Args({32});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({64});
BENCHMARK(BM_EXO_SAXPY) -> Args({64});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({128});
BENCHMARK(BM_EXO_SAXPY) -> Args({128});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({256});
BENCHMARK(BM_EXO_SAXPY) -> Args({256});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({512});
BENCHMARK(BM_EXO_SAXPY) -> Args({512});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({1024});
BENCHMARK(BM_EXO_SAXPY) -> Args({1024});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({2048});
BENCHMARK(BM_EXO_SAXPY) -> Args({2048});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({4096});
BENCHMARK(BM_EXO_SAXPY) -> Args({4096});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({8192});
BENCHMARK(BM_EXO_SAXPY) -> Args({8192});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({16384});
BENCHMARK(BM_EXO_SAXPY) -> Args({16384});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({32768});
BENCHMARK(BM_EXO_SAXPY) -> Args({32768});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({65536});
BENCHMARK(BM_EXO_SAXPY) -> Args({65536});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({131072});
BENCHMARK(BM_EXO_SAXPY) -> Args({131072});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({262144});
BENCHMARK(BM_EXO_SAXPY) -> Args({262144});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({524288});
BENCHMARK(BM_EXO_SAXPY) -> Args({524288});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({1048576});
BENCHMARK(BM_EXO_SAXPY) -> Args({1048576});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({2097152});
BENCHMARK(BM_EXO_SAXPY) -> Args({2097152});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({4194304});
BENCHMARK(BM_EXO_SAXPY) -> Args({4194304});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({8388608});
BENCHMARK(BM_EXO_SAXPY) -> Args({8388608});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({16777216});
BENCHMARK(BM_EXO_SAXPY) -> Args({16777216});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({33554432});
BENCHMARK(BM_EXO_SAXPY) -> Args({33554432});


BENCHMARK(BM_CBLAS_SAXPY) -> Args({67108864});
BENCHMARK(BM_EXO_SAXPY) -> Args({67108864});

