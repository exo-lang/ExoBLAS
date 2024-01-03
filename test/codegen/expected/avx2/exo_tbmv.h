
#pragma once
#ifndef EXO_TBMV_H
#define EXO_TBMV_H

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
struct exo_win_1f64{
    double * const data;
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
// exo_dtbmv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Lower_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Upper_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_dtbmv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag );

// exo_stbmv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Lower_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Upper_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );

// exo_stbmv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag );



#ifdef __cplusplus
}
#endif
#endif  // EXO_TBMV_H
