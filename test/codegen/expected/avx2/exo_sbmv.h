
#pragma once
#ifndef EXO_SBMV_H
#define EXO_SBMV_H

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
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f64c{
    const double * const data;
    const int_fast32_t strides[2];
};
// exo_dsbmv_row_major_Lower_stride_1(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_dsbmv_row_major_Lower_stride_any(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_dsbmv_row_major_Upper_stride_1(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_dsbmv_row_major_Upper_stride_any(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_dsbmv_scal_y_stride_1(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_scal_y_stride_1( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y );

// exo_dsbmv_scal_y_stride_any(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_scal_y_stride_any( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y );

// exo_ssbmv_row_major_Lower_stride_1(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_ssbmv_row_major_Lower_stride_any(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_ssbmv_row_major_Upper_stride_1(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_ssbmv_row_major_Upper_stride_any(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_ssbmv_scal_y_stride_1(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_scal_y_stride_1( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y );

// exo_ssbmv_scal_y_stride_any(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_scal_y_stride_any( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y );



#ifdef __cplusplus
}
#endif
#endif  // EXO_SBMV_H
