#include "exo_gemv.h"



#include <stdio.h>
#include <stdlib.h>



// exo_dgemv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[i * A.strides[0] + j];
  }
  y.data[i] = beta_ * y.data[i] + alpha_ * result;
}
}

// exo_dgemv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = beta_ * y.data[i * y.strides[0]] + alpha_ * result;
}
}

// exo_dgemv_row_major_Trans_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dgemv_row_major_Trans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k] = beta_ * y.data[k];
}
for (int_fast32_t i = 0; i < m; i++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[i];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}

// exo_dgemv_row_major_Trans_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dgemv_row_major_Trans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k * y.strides[0]] = beta_ * y.data[k * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[i * x.strides[0]];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j * y.strides[0]] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}

// exo_sgemv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[i * A.strides[0] + j];
  }
  y.data[i] = beta_ * y.data[i] + alpha_ * result;
}
}

// exo_sgemv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = beta_ * y.data[i * y.strides[0]] + alpha_ * result;
}
}

// exo_sgemv_row_major_Trans_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sgemv_row_major_Trans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k] = beta_ * y.data[k];
}
for (int_fast32_t i = 0; i < m; i++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[i];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}

// exo_sgemv_row_major_Trans_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sgemv_row_major_Trans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k * y.strides[0]] = beta_ * y.data[k * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[i * x.strides[0]];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j * y.strides[0]] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}

