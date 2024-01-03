
#pragma once
#ifndef EXO_GBMV_H
#define EXO_GBMV_H

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
// exo_dgbmv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     kl : size,
//     ku : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     a : [f64][m, ku + kl + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgbmv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, int_fast32_t kl, int_fast32_t ku, const double* alpha, const double* beta, struct exo_win_2f64c a, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_dgbmv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     kl : size,
//     ku : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     a : [f64][m, ku + kl + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgbmv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, int_fast32_t kl, int_fast32_t ku, const double* alpha, const double* beta, struct exo_win_2f64c a, struct exo_win_1f64c x, struct exo_win_1f64 y );

// exo_sgbmv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     kl : size,
//     ku : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     a : [f32][m, ku + kl + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgbmv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, int_fast32_t kl, int_fast32_t ku, const float* alpha, const float* beta, struct exo_win_2f32c a, struct exo_win_1f32c x, struct exo_win_1f32 y );

// exo_sgbmv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     kl : size,
//     ku : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     a : [f32][m, ku + kl + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgbmv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, int_fast32_t kl, int_fast32_t ku, const float* alpha, const float* beta, struct exo_win_2f32c a, struct exo_win_1f32c x, struct exo_win_1f32 y );



#ifdef __cplusplus
}
#endif
#endif  // EXO_GBMV_H
