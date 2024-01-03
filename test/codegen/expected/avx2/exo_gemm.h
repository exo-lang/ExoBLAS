
#pragma once
#ifndef EXO_GEMM_H
#define EXO_GEMM_H

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
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
// exo_sgemm_matmul_stride_any(
//     M : size,
//     N : size,
//     K : size,
//     A : [f32][M, K] @DRAM,
//     B : [f32][K, N] @DRAM,
//     C : [f32][M, N] @DRAM
// )
void exo_sgemm_matmul_stride_any( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C );



#ifdef __cplusplus
}
#endif
#endif  // EXO_GEMM_H
