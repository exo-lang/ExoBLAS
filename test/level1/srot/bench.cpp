#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_rot.h"

static void BM_CBLAS_SROT(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    float c = 2.342;
    float s = 1.23;

    for (auto _ : state) {
        cblas_srot(n, x.data(), 1, y.data(), 1, c, s);
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SROT(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    float c = 2.342;
    float s = 1.23;

    for (auto _ : state) {
        exo_srot(nullptr, n, exo_win_1f32{x.data(), {1}}, exo_win_1f32{y.data(), {1}}, &c, &s);
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SROT) -> Args({1});
BENCHMARK(BM_EXO_SROT) -> Args({1});


BENCHMARK(BM_CBLAS_SROT) -> Args({2});
BENCHMARK(BM_EXO_SROT) -> Args({2});


BENCHMARK(BM_CBLAS_SROT) -> Args({4});
BENCHMARK(BM_EXO_SROT) -> Args({4});


BENCHMARK(BM_CBLAS_SROT) -> Args({8});
BENCHMARK(BM_EXO_SROT) -> Args({8});


BENCHMARK(BM_CBLAS_SROT) -> Args({16});
BENCHMARK(BM_EXO_SROT) -> Args({16});


BENCHMARK(BM_CBLAS_SROT) -> Args({32});
BENCHMARK(BM_EXO_SROT) -> Args({32});


BENCHMARK(BM_CBLAS_SROT) -> Args({64});
BENCHMARK(BM_EXO_SROT) -> Args({64});


BENCHMARK(BM_CBLAS_SROT) -> Args({128});
BENCHMARK(BM_EXO_SROT) -> Args({128});


BENCHMARK(BM_CBLAS_SROT) -> Args({256});
BENCHMARK(BM_EXO_SROT) -> Args({256});


BENCHMARK(BM_CBLAS_SROT) -> Args({512});
BENCHMARK(BM_EXO_SROT) -> Args({512});


BENCHMARK(BM_CBLAS_SROT) -> Args({1024});
BENCHMARK(BM_EXO_SROT) -> Args({1024});


BENCHMARK(BM_CBLAS_SROT) -> Args({2048});
BENCHMARK(BM_EXO_SROT) -> Args({2048});


BENCHMARK(BM_CBLAS_SROT) -> Args({4096});
BENCHMARK(BM_EXO_SROT) -> Args({4096});


BENCHMARK(BM_CBLAS_SROT) -> Args({8192});
BENCHMARK(BM_EXO_SROT) -> Args({8192});


BENCHMARK(BM_CBLAS_SROT) -> Args({16384});
BENCHMARK(BM_EXO_SROT) -> Args({16384});


BENCHMARK(BM_CBLAS_SROT) -> Args({32768});
BENCHMARK(BM_EXO_SROT) -> Args({32768});


BENCHMARK(BM_CBLAS_SROT) -> Args({65536});
BENCHMARK(BM_EXO_SROT) -> Args({65536});


BENCHMARK(BM_CBLAS_SROT) -> Args({131072});
BENCHMARK(BM_EXO_SROT) -> Args({131072});


BENCHMARK(BM_CBLAS_SROT) -> Args({262144});
BENCHMARK(BM_EXO_SROT) -> Args({262144});


BENCHMARK(BM_CBLAS_SROT) -> Args({524288});
BENCHMARK(BM_EXO_SROT) -> Args({524288});


BENCHMARK(BM_CBLAS_SROT) -> Args({1048576});
BENCHMARK(BM_EXO_SROT) -> Args({1048576});


BENCHMARK(BM_CBLAS_SROT) -> Args({2097152});
BENCHMARK(BM_EXO_SROT) -> Args({2097152});


BENCHMARK(BM_CBLAS_SROT) -> Args({4194304});
BENCHMARK(BM_EXO_SROT) -> Args({4194304});


BENCHMARK(BM_CBLAS_SROT) -> Args({8388608});
BENCHMARK(BM_EXO_SROT) -> Args({8388608});


BENCHMARK(BM_CBLAS_SROT) -> Args({16777216});
BENCHMARK(BM_EXO_SROT) -> Args({16777216});


BENCHMARK(BM_CBLAS_SROT) -> Args({33554432});
BENCHMARK(BM_EXO_SROT) -> Args({33554432});


BENCHMARK(BM_CBLAS_SROT) -> Args({67108864});
BENCHMARK(BM_EXO_SROT) -> Args({67108864});

