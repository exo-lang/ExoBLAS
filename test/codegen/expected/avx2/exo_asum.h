
#pragma once
#ifndef EXO_ASUM_H
#define EXO_ASUM_H

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
// exo_dasum_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dasum_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result );

// exo_dasum_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dasum_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result );

// exo_sasum_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sasum_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result );

// exo_sasum_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sasum_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result );



#ifdef __cplusplus
}
#endif
#endif  // EXO_ASUM_H
