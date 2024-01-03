#include "exo_syr2.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dsyr2_row_major_Lower_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr2_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i + 1; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i] * y.data[j] + *alpha * y.data[i] * x.data[j];
  }
}
}

// exo_dsyr2_row_major_Lower_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr2_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i + 1; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * y.data[j * y.strides[0]] + *alpha * y.data[i * y.strides[0]] * x.data[j * x.strides[0]];
  }
}
}

// exo_dsyr2_row_major_Upper_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr2_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < n - i; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i] * y.data[i + j] + *alpha * y.data[i] * x.data[i + j];
  }
}
}

// exo_dsyr2_row_major_Upper_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr2_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64c y, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < n - i; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i * x.strides[0]] * y.data[(i + j) * y.strides[0]] + *alpha * y.data[i * y.strides[0]] * x.data[(i + j) * x.strides[0]];
  }
}
}

// exo_ssyr2_row_major_Lower_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr2_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i + 1; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i] * y.data[j] + *alpha * y.data[i] * x.data[j];
  }
}
}

// exo_ssyr2_row_major_Lower_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr2_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i + 1; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * y.data[j * y.strides[0]] + *alpha * y.data[i * y.strides[0]] * x.data[j * x.strides[0]];
  }
}
}

// exo_ssyr2_row_major_Upper_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr2_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < n - i; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i] * y.data[i + j] + *alpha * y.data[i] * x.data[i + j];
  }
}
}

// exo_ssyr2_row_major_Upper_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr2_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32c y, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < n - i; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i * x.strides[0]] * y.data[(i + j) * y.strides[0]] + *alpha * y.data[i * y.strides[0]] * x.data[(i + j) * x.strides[0]];
  }
}
}

