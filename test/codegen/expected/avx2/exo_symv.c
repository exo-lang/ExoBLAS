#include "exo_symv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dsymv_row_major_Lower_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    y.data[j] += temp * A.data[i * A.strides[0] + j];
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  y.data[i] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_dsymv_row_major_Lower_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i * x.strides[0]];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    y.data[j * y.strides[0]] += temp * A.data[i * A.strides[0] + j];
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_dsymv_row_major_Upper_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < n - i - 1; j++) {
    y.data[i + j + 1] += temp * A.data[i * A.strides[0] + i + j + 1];
    dot += A.data[i * A.strides[0] + i + j + 1] * x.data[i + j + 1];
  }
  y.data[i] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_dsymv_row_major_Upper_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i * x.strides[0]];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < n - i - 1; j++) {
    y.data[(i + j + 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + i + j + 1];
    dot += A.data[i * A.strides[0] + i + j + 1] * x.data[(i + j + 1) * x.strides[0]];
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_dsymv_scal_y_stride_1(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_scal_y_stride_1( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y ) {
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = *beta * y.data[i];
}
}

// exo_dsymv_scal_y_stride_any(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsymv_scal_y_stride_any( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
}

// exo_ssymv_row_major_Lower_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    y.data[j] += temp * A.data[i * A.strides[0] + j];
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  y.data[i] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_ssymv_row_major_Lower_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i * x.strides[0]];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    y.data[j * y.strides[0]] += temp * A.data[i * A.strides[0] + j];
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_ssymv_row_major_Upper_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < n - i - 1; j++) {
    y.data[i + j + 1] += temp * A.data[i * A.strides[0] + i + j + 1];
    dot += A.data[i * A.strides[0] + i + j + 1] * x.data[i + j + 1];
  }
  y.data[i] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_ssymv_row_major_Upper_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i * x.strides[0]];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < n - i - 1; j++) {
    y.data[(i + j + 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + i + j + 1];
    dot += A.data[i * A.strides[0] + i + j + 1] * x.data[(i + j + 1) * x.strides[0]];
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + i] + *alpha * dot;
}
}

// exo_ssymv_scal_y_stride_1(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_scal_y_stride_1( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y ) {
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = *beta * y.data[i];
}
}

// exo_ssymv_scal_y_stride_any(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssymv_scal_y_stride_any( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
}

