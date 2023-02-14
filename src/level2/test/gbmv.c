#include "gbmv.h"



#include <stdio.h>
#include <stdlib.h>



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
void gbmv_no_trans( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, const float* a, const float* x, float* y, int_fast32_t kl, int_fast32_t ku ) {
for (int i = 0; i < m; i++) {
  y[(i) * (1)] = *beta * y[(i) * (1)];
  for (int j = 0; j < n; j++) {
    if (i - kl <= j && j <= i + ku) {
      if (0 <= j + kl - i && j + kl - i < n) {
        y[(i) * (1)] += *alpha * a[(i) * (n) + (j + kl - i) * (1)] * x[(j) * (1)];
      }
    }
  }
}
}

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
void gbmv_trans( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, const float* a, const float* x, float* y, int_fast32_t kl, int_fast32_t ku ) {
for (int i = 0; i < m; i++) {
  y[(i) * (1)] = *beta * y[(i) * (1)];
  for (int j = 0; j < n; j++) {
    if (i - ku <= j && j <= i + kl) {
      if (0 <= j + kl - i && j + kl - i < n) {
        y[(i) * (1)] += *alpha * a[(j + kl - i) * (m) + (i) * (1)] * x[(j) * (1)];
      }
    }
  }
}
}

