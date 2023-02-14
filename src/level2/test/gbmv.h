
#pragma once
#ifndef GBMV_H
#define GBMV_H

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



// gbmv_no_trans(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     n : size,
//     m : size,
//     a : f32[m, n] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM,
//     kl : size,
//     ku : size
// )
void gbmv_no_trans( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, const float* a, const float* x, float* y, int_fast32_t kl, int_fast32_t ku );

// gbmv_trans(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     n : size,
//     m : size,
//     a : f32[n, m] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM,
//     kl : size,
//     ku : size
// )
void gbmv_trans( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, const float* a, const float* x, float* y, int_fast32_t kl, int_fast32_t ku );



#ifdef __cplusplus
}
#endif
#endif  // GBMV_H
