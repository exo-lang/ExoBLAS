#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_rotm.h"

static void BM_CBLAS_SROTM(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    auto H = generate1d_sbuffer(5, 1);
    H[0] = state.range(1);
    H[1] = 1.2;
    H[2] = 2.2;
    H[3] = 3.2;
    H[4] = 4.2;

    for (auto _ : state) {
        cblas_srotm(n, x.data(), 1, y.data(), 1, H.data());
    }

    // state.counters["flops"] = ;
}

static void BM_EXO_SROTM(benchmark::State& state) {
    auto n = state.range(0);

    std::vector<float> x = generate1d_sbuffer(n, 1);
    std::vector<float> y = generate1d_sbuffer(n, 1);
    auto H = generate1d_sbuffer(4, 1);
    H[0] = 1.2;
    H[1] = 3.2;
    H[2] = 2.2;
    H[3] = 4.2;

    for (auto _ : state) {
        exo_srotm(nullptr, n, exo_win_1f32{x.data(), {1}}, exo_win_1f32{y.data(), {1}}, state.range(1), H.data());
    }

    // state.counters["flops"] = ;
}

// Register the function as a benchmark
BENCHMARK(BM_CBLAS_SROTM)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}
    });
BENCHMARK(BM_EXO_SROTM)->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}
    });