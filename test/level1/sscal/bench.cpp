#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_scal.h"

static void BM_CBLAS_SSCAL(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    float alpha = 2.5;

    for (auto _ : state) {
        cblas_sscal(n, alpha, x.data(), 1);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SSCAL(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    float alpha = 2.5;

    for (auto _ : state) {
        exo_sscal(nullptr, n, &alpha, exo_win_1f32{.data = x.data(), .strides = {1}});
    }

    x = generate1d_sbuffer(n, 1);
    std::vector<float> expected = generate1d_sbuffer(n, 1);

    for (int i = 0; i < n; ++i) {
        expected[i] = alpha * x[i];
    }

    exo_sscal(nullptr, n, &alpha, exo_win_1f32{.data = x.data(), .strides = {1}});

    for (int i = 0; i < n; ++i) {
        if (x[i] != expected[i]) {
            printf("Failed ! got: %f, expected: %f, at index: %d\n", x[i], expected[i], i);
            exit(1);
        }
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SSCAL) -> Args({1});
BENCHMARK(BM_EXO_SSCAL) -> Args({1});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({2});
BENCHMARK(BM_EXO_SSCAL) -> Args({2});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({4});
BENCHMARK(BM_EXO_SSCAL) -> Args({4});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({8});
BENCHMARK(BM_EXO_SSCAL) -> Args({8});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({16});
BENCHMARK(BM_EXO_SSCAL) -> Args({16});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({32});
BENCHMARK(BM_EXO_SSCAL) -> Args({32});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({64});
BENCHMARK(BM_EXO_SSCAL) -> Args({64});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({128});
BENCHMARK(BM_EXO_SSCAL) -> Args({128});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({256});
BENCHMARK(BM_EXO_SSCAL) -> Args({256});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({512});
BENCHMARK(BM_EXO_SSCAL) -> Args({512});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({1024});
BENCHMARK(BM_EXO_SSCAL) -> Args({1024});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({2048});
BENCHMARK(BM_EXO_SSCAL) -> Args({2048});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({4096});
BENCHMARK(BM_EXO_SSCAL) -> Args({4096});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({8192});
BENCHMARK(BM_EXO_SSCAL) -> Args({8192});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({16384});
BENCHMARK(BM_EXO_SSCAL) -> Args({16384});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({32768});
BENCHMARK(BM_EXO_SSCAL) -> Args({32768});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({65536});
BENCHMARK(BM_EXO_SSCAL) -> Args({65536});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({131072});
BENCHMARK(BM_EXO_SSCAL) -> Args({131072});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({262144});
BENCHMARK(BM_EXO_SSCAL) -> Args({262144});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({524288});
BENCHMARK(BM_EXO_SSCAL) -> Args({524288});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({1048576});
BENCHMARK(BM_EXO_SSCAL) -> Args({1048576});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({2097152});
BENCHMARK(BM_EXO_SSCAL) -> Args({2097152});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({4194304});
BENCHMARK(BM_EXO_SSCAL) -> Args({4194304});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({8388608});
BENCHMARK(BM_EXO_SSCAL) -> Args({8388608});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({16777216});
BENCHMARK(BM_EXO_SSCAL) -> Args({16777216});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({33554432});
BENCHMARK(BM_EXO_SSCAL) -> Args({33554432});


BENCHMARK(BM_CBLAS_SSCAL) -> Args({67108864});
BENCHMARK(BM_EXO_SSCAL) -> Args({67108864});

