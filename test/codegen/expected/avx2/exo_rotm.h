
#pragma once
#ifndef EXO_ROTM_H
#define EXO_ROTM_H

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
// exo_drotm_flag_neg_one_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_neg_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_drotm_flag_neg_one_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_neg_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_drotm_flag_one_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_drotm_flag_one_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_drotm_flag_zero_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_zero_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_drotm_flag_zero_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_zero_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H );

// exo_srotm_flag_neg_one_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_neg_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );

// exo_srotm_flag_neg_one_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_neg_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );

// exo_srotm_flag_one_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );

// exo_srotm_flag_one_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );

// exo_srotm_flag_zero_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_zero_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );

// exo_srotm_flag_zero_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_zero_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H );



#ifdef __cplusplus
}
#endif
#endif  // EXO_ROTM_H
