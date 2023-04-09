#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_dsbmv.h"

static void BM_cblas_dsbmv(benchmark::State& state) {
    const int N = state.range(0);
    const int K = state.range(1);
    const enum CBLAS_ORDER order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const enum CBLAS_UPLO Uplo = state.range(3) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
    const double alpha = state.range(4);
    const int lda = state.range(5);
    const int incX = state.range(6);
    const double beta = state.range(7);
    const int incY = state.range(8); 
    size_t alignmentA = state.range(9);
    size_t alignmentX = state.range(10);
    size_t alignmentY = state.range(11);
    
    auto A = AlignedBuffer<double>(N * lda, 1, alignmentA);
    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        cblas_dsbmv(order, Uplo, N, K, alpha, A.data(), lda, X.data(), incX, beta, Y.data(), incY);
    }
}

static void BM_exo_dsbmv(benchmark::State& state) {
    const int N = state.range(0);
    const int K = state.range(1);
    const enum CBLAS_ORDER order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const enum CBLAS_UPLO Uplo = state.range(3) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
    const double alpha = state.range(4);
    const int lda = state.range(5);
    const int incX = state.range(6);
    const double beta = state.range(7);
    const int incY = state.range(8); 
    size_t alignmentA = state.range(9);
    size_t alignmentX = state.range(10);
    size_t alignmentY = state.range(11);
    
    auto A = AlignedBuffer<double>(N * lda, 1, alignmentA);
    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);

    for (auto _ : state) {
        exo_dsbmv(order, Uplo, N, K, alpha, A.data(), lda, X.data(), incX, beta, Y.data(), incY);
    }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark* b) {
    for (int order = 0; order < 1; ++order) {
        for (int Uplo = 0; Uplo <= 1; ++Uplo) {
            for (int alpha = 0; alpha <= 2; ++alpha) {
                for (int lda_diff = 0; lda_diff <= 1; ++lda_diff) {
                    for (int incX = 1; incX <= 1; ++incX) {
                        for (int beta = 0; beta <= 2; ++beta) {
                            for (int incY = 1; incY <= 1; ++incY) {
                                for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                                    for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                                        for (int alignmentY = 64; alignmentY <= 64; ++alignmentY) {
                                            for (int N = 1; N <= (1 << 10); N *= 2) {
                                                for (int K = N - 1; K >= 1; K /= 2) {
                                                    int lda = K + 1 + lda_diff;
                                                    b->Args({N, K, order, Uplo, alpha, lda, incX, beta, incY,
                                                    alignmentA, alignmentX, alignmentY});
                                                }
                                                b->Args({N, 0, order, Uplo, alpha, 1 + lda_diff, incX, beta, incY,
                                                    alignmentA, alignmentX, alignmentY});
                                            }
                                            for (int N = 7; N <= (1 << 10); N *= 7) {
                                                for (int K = N - 1; K >= 1; K /= 2) {
                                                    int lda = K + 1 + lda_diff;
                                                    b->Args({N, K, order, Uplo, alpha, lda, incX, beta, incY,
                                                    alignmentA, alignmentX, alignmentY});
                                                }
                                                b->Args({N, 0, order, Uplo, alpha, 1 + lda_diff, incX, beta, incY,
                                                    alignmentA, alignmentX, alignmentY});
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

BENCHMARK(BM_cblas_dsbmv)->ArgNames({"N", "K", "order", "Uplo", "alpha", "lda", "incX",
                                     "beta", "incY", "alignmentA", "alignmentX",
                                     "alignmentY"})->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_dsbmv)->ArgNames({"N", "K", "order", "Uplo", "alpha", "lda", "incX",
                                     "beta", "incY", "alignmentA", "alignmentX",
                                     "alignmentY"})->Apply(CustomArgumentsPacked);
