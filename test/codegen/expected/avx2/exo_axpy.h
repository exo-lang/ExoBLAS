
#pragma once
#ifndef EXO_AXPY_H
#define EXO_AXPY_H

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
// exo_daxpy_alpha_1_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_alpha_1_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_daxpy_alpha_1_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_alpha_1_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_daxpy_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_daxpy_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_saxpy_alpha_1_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_alpha_1_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_saxpy_alpha_1_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_alpha_1_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_saxpy_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_saxpy_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32 y );



#ifdef __cplusplus
}
#endif
#endif  // EXO_AXPY_H
