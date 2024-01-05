
#pragma once
#ifndef EXO_SAD_H
#define EXO_SAD_H

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



// sad(
//     x : i8[1024] @DRAM,
//     y : i8[1024] @DRAM,
//     result : f32 @DRAM
// )
void sad( void *ctxt, const int8_t* x, const int8_t* y, float* result );



#ifdef __cplusplus
}
#endif
#endif  // EXO_SAD_H
