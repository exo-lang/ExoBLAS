#include "exo_tbsv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dtbsv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + k - j - 1] * x.data[i - j - 1];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i] = (x.data[i] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(i + k) * A.strides[0] + k - j - 1] * x.data[i + k - j - 1];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(i + k) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i + k] = (x.data[i + k] - dot) / pivot;
}
}

// exo_dtbsv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + k - j - 1] * x.data[(i - j - 1) * x.strides[0]];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(i + k) * A.strides[0] + k - j - 1] * x.data[(i + k - j - 1) * x.strides[0]];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(i + k) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[(i + k) * x.strides[0]] = (x.data[(i + k) * x.strides[0]] - dot) / pivot;
}
}

// exo_dtbsv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (n - i + j < n) {
      dot += A.data[(n - i + j) * A.strides[0] + k - j - 1] * x.data[(n - i + j) * x.strides[0]];
    }
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_dtbsv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < k; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + j + 1] * x.data[n - i + j];
  }
  x.data[n - i - 1] = (x.data[n - i - 1] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - (k + i) - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - (k + i) - 1) * A.strides[0] + j + 1] * x.data[n - (k + i) + j];
  }
  x.data[n - (k + i) - 1] = (x.data[n - (k + i) - 1] - dot) / pivot;
}
}

// exo_dtbsv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < k; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + j + 1] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - (k + i) - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - (k + i) - 1) * A.strides[0] + j + 1] * x.data[(n - (k + i) + j) * x.strides[0]];
  }
  x.data[(n - (k + i) - 1) * x.strides[0]] = (x.data[(n - (k + i) - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_dtbsv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtbsv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      dot += A.data[(i - j - 1) * A.strides[0] + j + 1] * x.data[(i - j - 1) * x.strides[0]];
    }
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

// exo_stbsv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + k - j - 1] * x.data[i - j - 1];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i] = (x.data[i] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(i + k) * A.strides[0] + k - j - 1] * x.data[i + k - j - 1];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(i + k) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i + k] = (x.data[i + k] - dot) / pivot;
}
}

// exo_stbsv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + k - j - 1] * x.data[(i - j - 1) * x.strides[0]];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(i + k) * A.strides[0] + k - j - 1] * x.data[(i + k - j - 1) * x.strides[0]];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(i + k) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  x.data[(i + k) * x.strides[0]] = (x.data[(i + k) * x.strides[0]] - dot) / pivot;
}
}

// exo_stbsv_row_major_Lower_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + k];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (n - i + j < n) {
      dot += A.data[(n - i + j) * A.strides[0] + k - j - 1] * x.data[(n - i + j) * x.strides[0]];
    }
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_stbsv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < k; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + j + 1] * x.data[n - i + j];
  }
  x.data[n - i - 1] = (x.data[n - i - 1] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - (k + i) - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - (k + i) - 1) * A.strides[0] + j + 1] * x.data[n - (k + i) + j];
  }
  x.data[n - (k + i) - 1] = (x.data[n - (k + i) - 1] - dot) / pivot;
}
}

// exo_stbsv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < k; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + j + 1] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
for (int_fast32_t i = 0; i < n - k; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - (k + i) - 1) * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    dot += A.data[(n - (k + i) - 1) * A.strides[0] + j + 1] * x.data[(n - (k + i) + j) * x.strides[0]];
  }
  x.data[(n - (k + i) - 1) * x.strides[0]] = (x.data[(n - (k + i) - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_stbsv_row_major_Upper_Trans_stride_any(
//     n : size,
//     k : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_stbsv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, int_fast32_t k, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
EXO_ASSUME(k <= n - 1);
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < k; j++) {
    if (i - j - 1 >= 0) {
      dot += A.data[(i - j - 1) * A.strides[0] + j + 1] * x.data[(i - j - 1) * x.strides[0]];
    }
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0]];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

