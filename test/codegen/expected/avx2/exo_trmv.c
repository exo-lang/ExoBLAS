#include "exo_trmv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  xCopy[i] = dot + A.data[i * A.strides[0] + i] * x.data[i];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] = xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  xCopy[i] = dot + A.data[i * A.strides[0] + i] * x.data[i * x.strides[0]];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] = xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  xCopy[i] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] += xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  xCopy[i] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] += xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i];
  }
  xCopy[i] += A.data[i * A.strides[0] + i] * x.data[i];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i * x.strides[0]];
  }
  xCopy[i] += A.data[i * A.strides[0] + i] * x.data[i * x.strides[0]];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_Trans_Unit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Lower_Trans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] += xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Lower_Trans_Unit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Lower_Trans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] += xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - j + n];
  }
  xCopy[-1 - i + n] = dot + A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[-1 - i + n];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] = xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - j + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] = dot + A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[(-1 - i + n) * x.strides[0]];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] = xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - j + n];
  }
  xCopy[-1 - i + n] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] += xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  double dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - j + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] += xCopy[l];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - i + n];
  }
  xCopy[-1 - i + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[-1 - i + n];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - i + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[(-1 - i + n) * x.strides[0]];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_Trans_Unit_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Upper_Trans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - i + n];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] += xCopy[i];
}
free(xCopy);
}

// exo_dtrmv_row_major_Upper_Trans_Unit_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM,
//     Diag : size
// )
void exo_dtrmv_row_major_Upper_Trans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_2f64c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
double *xCopy = (double*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - i + n) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] += xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  xCopy[i] = dot + A.data[i * A.strides[0] + i] * x.data[i];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] = xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  xCopy[i] = dot + A.data[i * A.strides[0] + i] * x.data[i * x.strides[0]];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] = xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_NonTrans_Unit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Lower_NonTrans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j];
  }
  xCopy[i] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] += xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_NonTrans_Unit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Lower_NonTrans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[i * A.strides[0] + j] * x.data[j * x.strides[0]];
  }
  xCopy[i] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] += xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_Trans_NonUnit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Lower_Trans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i];
  }
  xCopy[i] += A.data[i * A.strides[0] + i] * x.data[i];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_Trans_NonUnit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Lower_Trans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i * x.strides[0]];
  }
  xCopy[i] += A.data[i * A.strides[0] + i] * x.data[i * x.strides[0]];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_Trans_Unit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Lower_Trans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] += xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Lower_Trans_Unit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Lower_Trans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[j] += A.data[i * A.strides[0] + j] * x.data[i * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] += xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - j + n];
  }
  xCopy[-1 - i + n] = dot + A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[-1 - i + n];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] = xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - j + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] = dot + A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[(-1 - i + n) * x.strides[0]];
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] = xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_NonTrans_Unit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Upper_NonTrans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - j + n];
  }
  xCopy[-1 - i + n] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l] += xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_NonTrans_Unit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_strmv_row_major_Upper_NonTrans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  float dot;
  dot = 0.0;
  for (int_fast32_t j = 0; j < i; j++) {
    dot += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - j + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] = dot;
}
for (int_fast32_t l = 0; l < n; l++) {
  x.data[l * x.strides[0]] += xCopy[l];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_Trans_NonUnit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Upper_Trans_NonUnit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - i + n];
  }
  xCopy[-1 - i + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[-1 - i + n];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] = xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_Trans_NonUnit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Upper_Trans_NonUnit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - i + n) * x.strides[0]];
  }
  xCopy[-1 - i + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - i + n] * x.data[(-1 - i + n) * x.strides[0]];
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_Trans_Unit_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Upper_Trans_Unit_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[-1 - i + n];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i] += xCopy[i];
}
free(xCopy);
}

// exo_strmv_row_major_Upper_Trans_Unit_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM,
//     Diag : size
// )
void exo_strmv_row_major_Upper_Trans_Unit_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_2f32c A, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
float *xCopy = (float*) malloc(n * sizeof(*xCopy));
for (int_fast32_t i = 0; i < n; i++) {
  xCopy[i] = 0.0;
}
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < i; j++) {
    xCopy[-1 - j + n] += A.data[(-1 - i + n) * A.strides[0] + -1 - j + n] * x.data[(-1 - i + n) * x.strides[0]];
  }
}
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] += xCopy[i];
}
free(xCopy);
}

