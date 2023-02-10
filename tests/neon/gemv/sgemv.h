
#pragma once
#ifndef SGEMV_H
#define SGEMV_H

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
// sgemv_exo(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     m : size,
//     n : size,
//     lda : size,
//     a : f32[m, lda] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM
// )
void sgemv_exo( void *ctxt, const float* alpha, const float* beta, int_fast32_t m, int_fast32_t n, int_fast32_t lda, const float* a, const float* x, float* y );

// sgemv_exo_v2(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     m : size,
//     n : size,
//     lda : size,
//     a : f32[m, lda] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM
// )
void sgemv_exo_v2( void *ctxt, const float* alpha, const float* beta, int_fast32_t m, int_fast32_t n, int_fast32_t lda, const float* a, const float* x, float* y );

// sgemv_transpose_exo(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     n : size,
//     m : size,
//     lda : size,
//     a : f32[n, lda] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM
// )
void sgemv_transpose_exo( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, int_fast32_t lda, const float* a, const float* x, float* y );



#ifdef __cplusplus
}
#endif
#endif  // SGEMV_H
