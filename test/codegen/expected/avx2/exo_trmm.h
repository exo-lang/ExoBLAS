
#pragma once
#ifndef EXO_TRMM_H
#define EXO_TRMM_H

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


struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
// exo_strmm_row_major_Left_Lower_NonTrans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Lower_NonTrans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag );

// exo_strmm_row_major_Left_Lower_Trans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Lower_Trans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag );

// exo_strmm_row_major_Left_Upper_NonTrans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Upper_NonTrans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag );

// exo_strmm_row_major_Left_Upper_Trans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Upper_Trans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag );



#ifdef __cplusplus
}
#endif
#endif  // EXO_TRMM_H
