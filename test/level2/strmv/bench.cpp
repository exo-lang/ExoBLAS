#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_strmv.h"

static void BM_cblas_strmv(benchmark::State& state) {
    const int N = state.range(0);
    const CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const CBLAS_UPLO Uplo = state.range(2) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
    const CBLAS_TRANSPOSE TransA = state.range(3) == 0 ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;
    const CBLAS_DIAG Diag = state.range(4) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
    const int lda = N + state.range(5);
    const int incX = state.range(6);
    const int alignmentX = state.range(7);
    const int alignmentA = state.range(8);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

    for (auto _ : state) {
        cblas_strmv(order, Uplo, TransA, Diag, N, A.data(), lda, X.data(), incX);
    }
}

static void BM_exo_strmv(benchmark::State& state) {
    const int N = state.range(0);
    const CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
    const CBLAS_UPLO Uplo = state.range(2) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
    const CBLAS_TRANSPOSE TransA = state.range(3) == 0 ? CBLAS_TRANSPOSE::CblasNoTrans : CBLAS_TRANSPOSE::CblasTrans;
    const CBLAS_DIAG Diag = state.range(4) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
    const int lda = N + state.range(5);
    const int incX = state.range(6);
    const int alignmentX = state.range(7);
    const int alignmentA = state.range(8);

    auto X = AlignedBuffer<float>(N, incX, alignmentX);
    auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

    for (auto _ : state) {
        exo_strmv(order, Uplo, TransA, Diag, N, A.data(), lda, X.data(), incX);
    }
}

BENCHMARK(BM_cblas_strmv)->ArgNames({"N", "order", "Uplo", "TransA", "Diag", "lda_diff", "incX", "alignmentX", "alignmentA"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 13), 2), {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, 7 * 7 * 7 * 7, 7), {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {64}, {64}
    });
BENCHMARK(BM_exo_strmv)->ArgNames({"N", "order", "Uplo", "TransA", "Diag", "lda_diff", "incX", "alignmentX", "alignmentA"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 13), 2), {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, 7 * 7 * 7 * 7, 7), {0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {1}, {64}, {64}
    });
