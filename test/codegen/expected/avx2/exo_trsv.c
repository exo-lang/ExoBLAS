#include "exo_trsv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dtrsv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i] = (x.data[i] - dot) / pivot;
}
}

// exo_dtrsv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

// exo_dtrsv_row_major_Lower_Trans_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i + j) * A.strides[0] + n - i - 1] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_dtrsv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + n - i + j] * x.data[n - i + j];
  }
  x.data[n - i - 1] = (x.data[n - i - 1] - dot) / pivot;
}
}

// exo_dtrsv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + n - i + j] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_dtrsv_row_major_Upper_Trans_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrsv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[j * A.strides[0] + i] * x.data[j * x.strides[0]];
  }
  double pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

// exo_strsv_row_major_Lower_NonTrans_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Lower_NonTrans_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i] = (x.data[i] - dot) / pivot;
}
}

// exo_strsv_row_major_Lower_NonTrans_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Lower_NonTrans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

// exo_strsv_row_major_Lower_Trans_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Lower_Trans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i + j) * A.strides[0] + n - i - 1] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_strsv_row_major_Upper_NonTrans_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Upper_NonTrans_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + n - i + j] * x.data[n - i + j];
  }
  x.data[n - i - 1] = (x.data[n - i - 1] - dot) / pivot;
}
}

// exo_strsv_row_major_Upper_NonTrans_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Upper_NonTrans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float pivot;
  if (Diag == 0) {
    pivot = A.data[(n - i - 1) * A.strides[0] + n - i - 1];
  } else {
    pivot = 1.0;
  }
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(n - i - 1) * A.strides[0] + n - i + j] * x.data[(n - i + j) * x.strides[0]];
  }
  x.data[(n - i - 1) * x.strides[0]] = (x.data[(n - i - 1) * x.strides[0]] - dot) / pivot;
}
}

// exo_strsv_row_major_Upper_Trans_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strsv_row_major_Upper_Trans_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[j * A.strides[0] + i] * x.data[j * x.strides[0]];
  }
  float pivot;
  if (Diag == 0) {
    pivot = A.data[i * A.strides[0] + i];
  } else {
    pivot = 1.0;
  }
  x.data[i * x.strides[0]] = (x.data[i * x.strides[0]] - dot) / pivot;
}
}

