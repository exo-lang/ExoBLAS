#include "exo_tbmv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dtbmv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[n - i - 1 - j - 1];
  }
  if (Diag == 0) {
    dot += x.data[n - i - 1] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    dot += x.data[n - i - 1];
  }
  x.data[n - i - 1] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[n - (n - k + i) - 1 - j - 1];
  }
  if (Diag == 0) {
    dot += x.data[n - (n - k + i) - 1] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    dot += x.data[n - (n - k + i) - 1];
  }
  x.data[n - (n - k + i) - 1] = dot;
}
}

// exo_dtbmv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[(n - i - 1 - j - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    dot += x.data[(n - i - 1) * x.strides[0]] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    dot += x.data[(n - i - 1) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[(n - (n - k + i) - 1 - j - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    dot += x.data[(n - (n - k + i) - 1) * x.strides[0]] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    dot += x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
  x.data[(n - (n - k + i) - 1) * x.strides[0]] = dot;
}
}

// exo_dtbmv_row_major_Lower_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
double *xRes = (double*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[n - i - 1 - j - 1] += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[n - i - 1];
  }
  if (Diag == 0) {
    xRes[n - i - 1] += x.data[n - i - 1] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    xRes[n - i - 1] += x.data[n - i - 1];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - (n - k + i) - 1 - j - 1] += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[n - (n - k + i) - 1];
  }
  if (Diag == 0) {
    xRes[n - (n - k + i) - 1] += x.data[n - (n - k + i) - 1] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    xRes[n - (n - k + i) - 1] += x.data[n - (n - k + i) - 1];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xRes[i];
}
free(xRes);
}

// exo_dtbmv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
double *xRes = (double*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[n - i - 1 - j - 1] += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[(n - i - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    xRes[n - i - 1] += x.data[(n - i - 1) * x.strides[0]] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    xRes[n - i - 1] += x.data[(n - i - 1) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - (n - k + i) - 1 - j - 1] += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    xRes[n - (n - k + i) - 1] += x.data[(n - (n - k + i) - 1) * x.strides[0]] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    xRes[n - (n - k + i) - 1] += x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xRes[i];
}
free(xRes);
}

// exo_dtbmv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[i] * A.data[i * A.strides[0]];
  } else {
    dot = x.data[i];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[i * A.strides[0] + j + 1] * x.data[i + j + 1];
  }
  x.data[i] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[n - k + i] * A.data[(n - k + i) * A.strides[0]];
  } else {
    dot = x.data[n - k + i];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[n - k + i + j + 1];
  }
  x.data[n - k + i] = dot;
}
}

// exo_dtbmv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[i * x.strides[0]] * A.data[i * A.strides[0]];
  } else {
    dot = x.data[i * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[i * A.strides[0] + j + 1] * x.data[(i + j + 1) * x.strides[0]];
  }
  x.data[i * x.strides[0]] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[(n - k + i) * x.strides[0]] * A.data[(n - k + i) * A.strides[0]];
  } else {
    dot = x.data[(n - k + i) * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[(n - k + i + j + 1) * x.strides[0]];
  }
  x.data[(n - k + i) * x.strides[0]] = dot;
}
}

// exo_dtbmv_row_major_Upper_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
double *xRes = (double*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  if (Diag == 0) {
    xRes[i] += x.data[i] * A.data[i * A.strides[0]];
  } else {
    xRes[i] += x.data[i];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[i + j + 1] += A.data[i * A.strides[0] + j + 1] * x.data[i];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  if (Diag == 0) {
    xRes[n - k + i] += x.data[n - k + i] * A.data[(n - k + i) * A.strides[0]];
  } else {
    xRes[n - k + i] += x.data[n - k + i];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - k + i + j + 1] += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[n - k + i];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xRes[i];
}
free(xRes);
}

// exo_dtbmv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_dtbmv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
double *xRes = (double*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  if (Diag == 0) {
    xRes[i] += x.data[i * x.strides[0]] * A.data[i * A.strides[0]];
  } else {
    xRes[i] += x.data[i * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[i + j + 1] += A.data[i * A.strides[0] + j + 1] * x.data[i * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  if (Diag == 0) {
    xRes[n - k + i] += x.data[(n - k + i) * x.strides[0]] * A.data[(n - k + i) * A.strides[0]];
  } else {
    xRes[n - k + i] += x.data[(n - k + i) * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - k + i + j + 1] += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[(n - k + i) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xRes[i];
}
free(xRes);
}

// exo_stbmv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[n - i - 1 - j - 1];
  }
  if (Diag == 0) {
    dot += x.data[n - i - 1] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    dot += x.data[n - i - 1];
  }
  x.data[n - i - 1] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[n - (n - k + i) - 1 - j - 1];
  }
  if (Diag == 0) {
    dot += x.data[n - (n - k + i) - 1] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    dot += x.data[n - (n - k + i) - 1];
  }
  x.data[n - (n - k + i) - 1] = dot;
}
}

// exo_stbmv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[(n - i - 1 - j - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    dot += x.data[(n - i - 1) * x.strides[0]] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    dot += x.data[(n - i - 1) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[(n - (n - k + i) - 1 - j - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    dot += x.data[(n - (n - k + i) - 1) * x.strides[0]] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    dot += x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
  x.data[(n - (n - k + i) - 1) * x.strides[0]] = dot;
}
}

// exo_stbmv_row_major_Lower_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
float *xRes = (float*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[n - i - 1 - j - 1] += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[n - i - 1];
  }
  if (Diag == 0) {
    xRes[n - i - 1] += x.data[n - i - 1] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    xRes[n - i - 1] += x.data[n - i - 1];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - (n - k + i) - 1 - j - 1] += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[n - (n - k + i) - 1];
  }
  if (Diag == 0) {
    xRes[n - (n - k + i) - 1] += x.data[n - (n - k + i) - 1] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    xRes[n - (n - k + i) - 1] += x.data[n - (n - k + i) - 1];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xRes[i];
}
free(xRes);
}

// exo_stbmv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
float *xRes = (float*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[n - i - 1 - j - 1] += A.data[(n - i - 1) * A.strides[0] + k - j - 1] * x.data[(n - i - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    xRes[n - i - 1] += x.data[(n - i - 1) * x.strides[0]] * A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    xRes[n - i - 1] += x.data[(n - i - 1) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - (n - k + i) - 1 - j - 1] += A.data[(n - (n - k + i) - 1) * A.strides[0] + k - j - 1] * x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
  if (Diag == 0) {
    xRes[n - (n - k + i) - 1] += x.data[(n - (n - k + i) - 1) * x.strides[0]] * A.data[(n - (n - k + i) - 1) * A.strides[0] + k];
  } else {
    xRes[n - (n - k + i) - 1] += x.data[(n - (n - k + i) - 1) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xRes[i];
}
free(xRes);
}

// exo_stbmv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[i] * A.data[i * A.strides[0]];
  } else {
    dot = x.data[i];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[i * A.strides[0] + j + 1] * x.data[i + j + 1];
  }
  x.data[i] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[n - k + i] * A.data[(n - k + i) * A.strides[0]];
  } else {
    dot = x.data[n - k + i];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[n - k + i + j + 1];
  }
  x.data[n - k + i] = dot;
}
}

// exo_stbmv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[i * x.strides[0]] * A.data[i * A.strides[0]];
  } else {
    dot = x.data[i * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[i * A.strides[0] + j + 1] * x.data[(i + j + 1) * x.strides[0]];
  }
  x.data[i * x.strides[0]] = dot;
}
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  if (Diag == 0) {
    dot = x.data[(n - k + i) * x.strides[0]] * A.data[(n - k + i) * A.strides[0]];
  } else {
    dot = x.data[(n - k + i) * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    dot += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[(n - k + i + j + 1) * x.strides[0]];
  }
  x.data[(n - k + i) * x.strides[0]] = dot;
}
}

// exo_stbmv_row_major_Upper_Trans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_Trans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
float *xRes = (float*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  if (Diag == 0) {
    xRes[i] += x.data[i] * A.data[i * A.strides[0]];
  } else {
    xRes[i] += x.data[i];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[i + j + 1] += A.data[i * A.strides[0] + j + 1] * x.data[i];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  if (Diag == 0) {
    xRes[n - k + i] += x.data[n - k + i] * A.data[(n - k + i) * A.strides[0]];
  } else {
    xRes[n - k + i] += x.data[n - k + i];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - k + i + j + 1] += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[n - k + i];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xRes[i];
}
free(xRes);
}

// exo_stbmv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, k + 1] @DRAM,
//     Diag : size
// )
void exo_stbmv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
float *xRes = (float*) malloc(n * sizeof(*xRes));
for (int_fast32_t i = 0; i < n; i++) {
  xRes[i] = 0.0;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  if (Diag == 0) {
    xRes[i] += x.data[i * x.strides[0]] * A.data[i * A.strides[0]];
  } else {
    xRes[i] += x.data[i * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k; j++) {
    xRes[i + j + 1] += A.data[i * A.strides[0] + j + 1] * x.data[i * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < k; i++) {
  if (Diag == 0) {
    xRes[n - k + i] += x.data[(n - k + i) * x.strides[0]] * A.data[(n - k + i) * A.strides[0]];
  } else {
    xRes[n - k + i] += x.data[(n - k + i) * x.strides[0]];
  }
  for (int_fast32_t j = 0; j < k - i - 1; j++) {
    xRes[n - k + i + j + 1] += A.data[(n - k + i) * A.strides[0] + j + 1] * x.data[(n - k + i) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xRes[i];
}
free(xRes);
}

