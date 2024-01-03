
#pragma once
#ifndef EXO_SYRK_H
#define EXO_SYRK_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f64{
    double * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f64c{
    const double * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f64{
    double * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f64c{
    const double * const data;
    const int_fast32_t strides[2];
};
// exo_dsyrk_lower_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_lower_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_dsyrk_upper_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C );

// exo_ssyrk_lower_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_lower_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );

// exo_ssyrk_upper_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C );



#ifdef __cplusplus
}
#endif
#endif  // EXO_SYRK_H
