#include "exo_sbmv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dsbmv_row_major_Lower_stride_1(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      y.data[i - j - 1] += temp * A.data[i * A.strides[0] + k - j - 1];
      dot += A.data[i * A.strides[0] + k - j - 1] * x.data[i - j - 1];
    }
  }
  y.data[i] += temp * A.data[i * A.strides[0] + k] + *alpha * dot;
}
}

// exo_dsbmv_row_major_Lower_stride_any(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i * x.strides[0]];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      y.data[(i - j - 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + k - j - 1];
      dot += A.data[i * A.strides[0] + k - j - 1] * x.data[(i - j - 1) * x.strides[0]];
    }
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + k] + *alpha * dot;
}
}

// exo_dsbmv_row_major_Upper_stride_1(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i + j + 1 < n) {
      y.data[i + j + 1] += temp * A.data[i * A.strides[0] + j + 1];
      dot += A.data[i * A.strides[0] + j + 1] * x.data[i + j + 1];
    }
  }
  y.data[i] += temp * A.data[i * A.strides[0]] + *alpha * dot;
}
}

// exo_dsbmv_row_major_Upper_stride_any(
//     n : size,
//     k : size,
//     alpha : f64 @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const double* alpha, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  double temp;
  temp = *alpha * x.data[i * x.strides[0]];
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i + j + 1 < n) {
      y.data[(i + j + 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + j + 1];
      dot += A.data[i * A.strides[0] + j + 1] * x.data[(i + j + 1) * x.strides[0]];
    }
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0]] + *alpha * dot;
}
}

// exo_dsbmv_scal_y_stride_1(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_scal_y_stride_1( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y ) {
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = *beta * y.data[i];
}
}

// exo_dsbmv_scal_y_stride_any(
//     n : size,
//     beta : f64 @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dsbmv_scal_y_stride_any( void *ctxt, int_fast32_t n, const double* beta, struct exo_win_1f64 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
}

// exo_ssbmv_row_major_Lower_stride_1(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      y.data[i - j - 1] += temp * A.data[i * A.strides[0] + k - j - 1];
      dot += A.data[i * A.strides[0] + k - j - 1] * x.data[i - j - 1];
    }
  }
  y.data[i] += temp * A.data[i * A.strides[0] + k] + *alpha * dot;
}
}

// exo_ssbmv_row_major_Lower_stride_any(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i * x.strides[0]];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      y.data[(i - j - 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + k - j - 1];
      dot += A.data[i * A.strides[0] + k - j - 1] * x.data[(i - j - 1) * x.strides[0]];
    }
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0] + k] + *alpha * dot;
}
}

// exo_ssbmv_row_major_Upper_stride_1(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i + j + 1 < n) {
      y.data[i + j + 1] += temp * A.data[i * A.strides[0] + j + 1];
      dot += A.data[i * A.strides[0] + j + 1] * x.data[i + j + 1];
    }
  }
  y.data[i] += temp * A.data[i * A.strides[0]] + *alpha * dot;
}
}

// exo_ssbmv_row_major_Upper_stride_any(
//     n : size,
//     k : size,
//     alpha : f32 @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, const float* alpha, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  float temp;
  temp = *alpha * x.data[i * x.strides[0]];
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i + j + 1 < n) {
      y.data[(i + j + 1) * y.strides[0]] += temp * A.data[i * A.strides[0] + j + 1];
      dot += A.data[i * A.strides[0] + j + 1] * x.data[(i + j + 1) * x.strides[0]];
    }
  }
  y.data[i * y.strides[0]] += temp * A.data[i * A.strides[0]] + *alpha * dot;
}
}

// exo_ssbmv_scal_y_stride_1(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_scal_y_stride_1( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y ) {
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = *beta * y.data[i];
}
}

// exo_ssbmv_scal_y_stride_any(
//     n : size,
//     beta : f32 @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_ssbmv_scal_y_stride_any( void *ctxt, int_fast32_t n, const float* beta, struct exo_win_1f32 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
}

