#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_ssyr.h"

static void BM_cblas_ssyr(benchmark::State& state) {
    const int N = state.range(0);
    const enum CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const enum CBLAS_UPLO Uplo = state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
    const float alpha = state.range(3);
    const int lda = state.range(4);
    const int incX = state.range(5);
    size_t alignmentA = state.range(6);
    size_t alignmentX = state.range(7);
    
    auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);
    auto X = AlignedBuffer<float>(N, incX, alignmentX);

    for (auto _ : state) {
        cblas_ssyr(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
    }
}

static void BM_exo_ssyr(benchmark::State& state) {
    const int N = state.range(0);
    const enum CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const enum CBLAS_UPLO Uplo = state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
    const float alpha = state.range(3);
    const int lda = state.range(4);
    const int incX = state.range(5);
    size_t alignmentA = state.range(6);
    size_t alignmentX = state.range(7);
    
    auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);
    auto X = AlignedBuffer<float>(N, incX, alignmentX);

    for (auto _ : state) {
        exo_ssyr(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
    }
}


static void CustomArgumentsPacked(benchmark::internal::Benchmark* b) {
    for (int order = 0; order < 1; ++order) {
        for (int Uplo = 0; Uplo <= 1; ++Uplo) {
            for (int alpha = 0; alpha <= 2; ++alpha) {
                for (int lda_diff = 0; lda_diff < 1; ++lda_diff) {
                    for (int incX = 1; incX <= 1; ++incX) {
                        for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                            for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                                for (int N = 1; N <= (1 << 10); N *= 2) {
                                    int lda = N + lda_diff;
                                    b->Args({N, order, Uplo, alpha, lda, incX,
                                    alignmentA, alignmentX});
                                }
                                for (int N = 7; N <= (1 << 10); N *= 7) {
                                    int lda = N + lda_diff;
                                    b->Args({N, order, Uplo, alpha, lda, incX,
                                    alignmentA, alignmentX});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

BENCHMARK(BM_cblas_ssyr)->ArgNames({"N", "order", "Uplo", "alpha", "lda", "incX",
                                     "alignmentA", "alignmentX"})->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_ssyr)->ArgNames({"N", "order", "Uplo", "alpha", "lda", "incX",
                                     "alignmentA", "alignmentX"})->Apply(CustomArgumentsPacked);
