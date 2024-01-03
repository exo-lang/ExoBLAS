#include "exo_ger.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dger_row_major_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][m, n] @DRAM
// )
void exo_dger_row_major_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  for (int_fast32_t j = 0; j < n; j++) {
    A.data[i * A.strides[0] + j] += alpha_ * x.data[i] * y.data[j];
  }
}
}

// exo_dger_row_major_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][m, n] @DRAM
// )
void exo_dger_row_major_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < m; i++) {
  for (int_fast32_t j = 0; j < n; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * y.data[j * y.strides[0]];
  }
}
}

// exo_sger_row_major_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][m, n] @DRAM
// )
void exo_sger_row_major_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  for (int_fast32_t j = 0; j < n; j++) {
    A.data[i * A.strides[0] + j] += alpha_ * x.data[i] * y.data[j];
  }
}
}

// exo_sger_row_major_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][m, n] @DRAM
// )
void exo_sger_row_major_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < m; i++) {
  for (int_fast32_t j = 0; j < n; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * y.data[j * y.strides[0]];
  }
}
}

