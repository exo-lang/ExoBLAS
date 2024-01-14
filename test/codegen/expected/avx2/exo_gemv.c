#include "exo_gemv.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction..."
avx2_reg_copy_pd(dst,src)
{dst_data} = {src_data};
*/

/* relying on the following instruction..."
avx2_reg_copy_ps(dst,src)
{dst_data} = {src_data};
*/
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
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < m; i++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[i * A.strides[0] + j];
  }
  y.data[i] = *beta * y.data[i] + *alpha * result;
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
for (int_fast32_t i = 0; i < m; i++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]] + *alpha * result;
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
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = beta_ * y.data[i];
}
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  static double alphaXi[4];
  alphaXi[0] = alpha_ * x.data[4 * io];
  alphaXi[1] = alpha_ * x.data[1 + 4 * io];
  alphaXi[2] = alpha_ * x.data[2 + 4 * io];
  alphaXi[3] = alpha_ * x.data[3 + 4 * io];
  for (int_fast32_t joo = 0; joo < ((n) / (8)); joo++) {
    __m256d yReg[2];
    yReg[0] = _mm256_loadu_pd(&y.data[8 * joo]);
    yReg[1] = _mm256_loadu_pd(&y.data[4 + 8 * joo]);
    __m256d var0[2];
    __m256d var1[2];
    __m256d var2[2];
    var0[0] = _mm256_broadcast_sd(&alphaXi[0]);
    var0[1] = _mm256_broadcast_sd(&alphaXi[0]);
    var1[0] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 * joo]);
    var1[1] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var2[0] = yReg[0];
    var2[1] = yReg[1];
    var2[0] = _mm256_fmadd_pd(var0[0], var1[0], var2[0]);
    var2[1] = _mm256_fmadd_pd(var0[1], var1[1], var2[1]);
    yReg[0] = var2[0];
    yReg[1] = var2[1];
    __m256d var0_1[2];
    __m256d var1_1[2];
    __m256d var2_1[2];
    var0_1[0] = _mm256_broadcast_sd(&alphaXi[1]);
    var0_1[1] = _mm256_broadcast_sd(&alphaXi[1]);
    var1_1[0] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var1_1[1] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var2_1[0] = yReg[0];
    var2_1[1] = yReg[1];
    var2_1[0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], var2_1[0]);
    var2_1[1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], var2_1[1]);
    yReg[0] = var2_1[0];
    yReg[1] = var2_1[1];
    __m256d var0_2[2];
    __m256d var1_2[2];
    __m256d var2_2[2];
    var0_2[0] = _mm256_broadcast_sd(&alphaXi[2]);
    var0_2[1] = _mm256_broadcast_sd(&alphaXi[2]);
    var1_2[0] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var1_2[1] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var2_2[0] = yReg[0];
    var2_2[1] = yReg[1];
    var2_2[0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], var2_2[0]);
    var2_2[1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], var2_2[1]);
    yReg[0] = var2_2[0];
    yReg[1] = var2_2[1];
    __m256d var0_3[2];
    __m256d var1_3[2];
    __m256d var2_3[2];
    var0_3[0] = _mm256_broadcast_sd(&alphaXi[3]);
    var0_3[1] = _mm256_broadcast_sd(&alphaXi[3]);
    var1_3[0] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var1_3[1] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var2_3[0] = yReg[0];
    var2_3[1] = yReg[1];
    var2_3[0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], var2_3[0]);
    var2_3[1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], var2_3[1]);
    yReg[0] = var2_3[0];
    yReg[1] = var2_3[1];
    _mm256_storeu_pd(&y.data[8 * joo], yReg[0]);
    _mm256_storeu_pd(&y.data[4 + 8 * joo], yReg[1]);
  }
  for (int_fast32_t joi = 0; joi < ((n) / (4)) % 2; joi++) {
    __m256d var0;
    var0 = _mm256_broadcast_sd(&alphaXi[0]);
    __m256d var1;
    var1 = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 * (n / 8) + 4 * joi]);
    __m256d var2;
    var2 = _mm256_loadu_pd(&y.data[8 * (n / 8) + 4 * joi]);
    var2 = _mm256_fmadd_pd(var0, var1, var2);
    _mm256_storeu_pd(&y.data[8 * (n / 8) + 4 * joi], var2);
    __m256d var0_1;
    var0_1 = _mm256_broadcast_sd(&alphaXi[1]);
    __m256d var1_1;
    var1_1 = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * (n / 8) + 4 * joi]);
    __m256d var2_1;
    var2_1 = _mm256_loadu_pd(&y.data[8 * (n / 8) + 4 * joi]);
    var2_1 = _mm256_fmadd_pd(var0_1, var1_1, var2_1);
    _mm256_storeu_pd(&y.data[8 * (n / 8) + 4 * joi], var2_1);
    __m256d var0_2;
    var0_2 = _mm256_broadcast_sd(&alphaXi[2]);
    __m256d var1_2;
    var1_2 = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * (n / 8) + 4 * joi]);
    __m256d var2_2;
    var2_2 = _mm256_loadu_pd(&y.data[8 * (n / 8) + 4 * joi]);
    var2_2 = _mm256_fmadd_pd(var0_2, var1_2, var2_2);
    _mm256_storeu_pd(&y.data[8 * (n / 8) + 4 * joi], var2_2);
    __m256d var0_3;
    var0_3 = _mm256_broadcast_sd(&alphaXi[3]);
    __m256d var1_3;
    var1_3 = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * (n / 8) + 4 * joi]);
    __m256d var2_3;
    var2_3 = _mm256_loadu_pd(&y.data[8 * (n / 8) + 4 * joi]);
    var2_3 = _mm256_fmadd_pd(var0_3, var1_3, var2_3);
    _mm256_storeu_pd(&y.data[8 * (n / 8) + 4 * joi], var2_3);
  }
  for (int_fast32_t ji = 0; ji < n % 4; ji++) {
    y.data[ji + (n / 4) * 4] += alphaXi[0] * A.data[4 * io * A.strides[0] + ji + (n / 4) * 4];
    y.data[ji + (n / 4) * 4] += alphaXi[1] * A.data[(1 + 4 * io) * A.strides[0] + ji + (n / 4) * 4];
    y.data[ji + (n / 4) * 4] += alphaXi[2] * A.data[(2 + 4 * io) * A.strides[0] + ji + (n / 4) * 4];
    y.data[ji + (n / 4) * 4] += alphaXi[3] * A.data[(3 + 4 * io) * A.strides[0] + ji + (n / 4) * 4];
  }
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 4) * 4];
  for (int_fast32_t joo = 0; joo < ((n) / (8)); joo++) {
    __m256d var0[2];
    __m256d var1[2];
    __m256d var2[2];
    var0[0] = _mm256_broadcast_sd((&alphaXi));
    var0[1] = _mm256_broadcast_sd((&alphaXi));
    var1[0] = _mm256_loadu_pd(&A.data[(4 * (m / 4) + ii) * (A.strides[0]) + 8 * joo]);
    var1[1] = _mm256_loadu_pd(&A.data[(4 * (m / 4) + ii) * (A.strides[0]) + 4 + 8 * joo]);
    var2[0] = _mm256_loadu_pd(&y.data[8 * joo]);
    var2[1] = _mm256_loadu_pd(&y.data[4 + 8 * joo]);
    var2[0] = _mm256_fmadd_pd(var0[0], var1[0], var2[0]);
    var2[1] = _mm256_fmadd_pd(var0[1], var1[1], var2[1]);
    _mm256_storeu_pd(&y.data[8 * joo], var2[0]);
    _mm256_storeu_pd(&y.data[4 + 8 * joo], var2[1]);
  }
  for (int_fast32_t joi = 0; joi < ((n) / (4)) % 2; joi++) {
    __m256d var0;
    var0 = _mm256_broadcast_sd((&alphaXi));
    __m256d var1;
    var1 = _mm256_loadu_pd(&A.data[(4 * (m / 4) + ii) * (A.strides[0]) + 8 * (n / 8) + 4 * joi]);
    __m256d var2;
    var2 = _mm256_loadu_pd(&y.data[8 * (n / 8) + 4 * joi]);
    var2 = _mm256_fmadd_pd(var0, var1, var2);
    _mm256_storeu_pd(&y.data[8 * (n / 8) + 4 * joi], var2);
  }
  for (int_fast32_t ji = 0; ji < n % 4; ji++) {
    y.data[ji + (n / 4) * 4] += alphaXi * A.data[(ii + (m / 4) * 4) * A.strides[0] + ji + (n / 4) * 4];
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
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  double alphaXi;
  alphaXi = *alpha * x.data[i * x.strides[0]];
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
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t i = 0; i < m; i++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[i * A.strides[0] + j];
  }
  y.data[i] = *beta * y.data[i] + *alpha * result;
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
for (int_fast32_t i = 0; i < m; i++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]] + *alpha * result;
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
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i] = beta_ * y.data[i];
}
for (int_fast32_t io = 0; io < ((m) / (8)); io++) {
  static float alphaXi[8];
  alphaXi[0] = alpha_ * x.data[8 * io];
  alphaXi[1] = alpha_ * x.data[1 + 8 * io];
  alphaXi[2] = alpha_ * x.data[2 + 8 * io];
  alphaXi[3] = alpha_ * x.data[3 + 8 * io];
  alphaXi[4] = alpha_ * x.data[4 + 8 * io];
  alphaXi[5] = alpha_ * x.data[5 + 8 * io];
  alphaXi[6] = alpha_ * x.data[6 + 8 * io];
  alphaXi[7] = alpha_ * x.data[7 + 8 * io];
  for (int_fast32_t joo = 0; joo < ((n) / (16)); joo++) {
    __m256 yReg[2];
    yReg[0] = _mm256_loadu_ps(&y.data[16 * joo]);
    yReg[1] = _mm256_loadu_ps(&y.data[8 + 16 * joo]);
    __m256 var0[2];
    __m256 var1[2];
    __m256 var2[2];
    var0[0] = _mm256_broadcast_ss(&alphaXi[0]);
    var0[1] = _mm256_broadcast_ss(&alphaXi[0]);
    var1[0] = _mm256_loadu_ps(&A.data[(8 * io) * (A.strides[0]) + 16 * joo]);
    var1[1] = _mm256_loadu_ps(&A.data[(8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2[0] = yReg[0];
    var2[1] = yReg[1];
    var2[0] = _mm256_fmadd_ps(var0[0], var1[0], var2[0]);
    var2[1] = _mm256_fmadd_ps(var0[1], var1[1], var2[1]);
    yReg[0] = var2[0];
    yReg[1] = var2[1];
    __m256 var0_1[2];
    __m256 var1_1[2];
    __m256 var2_1[2];
    var0_1[0] = _mm256_broadcast_ss(&alphaXi[1]);
    var0_1[1] = _mm256_broadcast_ss(&alphaXi[1]);
    var1_1[0] = _mm256_loadu_ps(&A.data[(1 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_1[1] = _mm256_loadu_ps(&A.data[(1 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_1[0] = yReg[0];
    var2_1[1] = yReg[1];
    var2_1[0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], var2_1[0]);
    var2_1[1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], var2_1[1]);
    yReg[0] = var2_1[0];
    yReg[1] = var2_1[1];
    __m256 var0_2[2];
    __m256 var1_2[2];
    __m256 var2_2[2];
    var0_2[0] = _mm256_broadcast_ss(&alphaXi[2]);
    var0_2[1] = _mm256_broadcast_ss(&alphaXi[2]);
    var1_2[0] = _mm256_loadu_ps(&A.data[(2 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_2[1] = _mm256_loadu_ps(&A.data[(2 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_2[0] = yReg[0];
    var2_2[1] = yReg[1];
    var2_2[0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], var2_2[0]);
    var2_2[1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], var2_2[1]);
    yReg[0] = var2_2[0];
    yReg[1] = var2_2[1];
    __m256 var0_3[2];
    __m256 var1_3[2];
    __m256 var2_3[2];
    var0_3[0] = _mm256_broadcast_ss(&alphaXi[3]);
    var0_3[1] = _mm256_broadcast_ss(&alphaXi[3]);
    var1_3[0] = _mm256_loadu_ps(&A.data[(3 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_3[1] = _mm256_loadu_ps(&A.data[(3 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_3[0] = yReg[0];
    var2_3[1] = yReg[1];
    var2_3[0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], var2_3[0]);
    var2_3[1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], var2_3[1]);
    yReg[0] = var2_3[0];
    yReg[1] = var2_3[1];
    __m256 var0_4[2];
    __m256 var1_4[2];
    __m256 var2_4[2];
    var0_4[0] = _mm256_broadcast_ss(&alphaXi[4]);
    var0_4[1] = _mm256_broadcast_ss(&alphaXi[4]);
    var1_4[0] = _mm256_loadu_ps(&A.data[(4 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_4[1] = _mm256_loadu_ps(&A.data[(4 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_4[0] = yReg[0];
    var2_4[1] = yReg[1];
    var2_4[0] = _mm256_fmadd_ps(var0_4[0], var1_4[0], var2_4[0]);
    var2_4[1] = _mm256_fmadd_ps(var0_4[1], var1_4[1], var2_4[1]);
    yReg[0] = var2_4[0];
    yReg[1] = var2_4[1];
    __m256 var0_5[2];
    __m256 var1_5[2];
    __m256 var2_5[2];
    var0_5[0] = _mm256_broadcast_ss(&alphaXi[5]);
    var0_5[1] = _mm256_broadcast_ss(&alphaXi[5]);
    var1_5[0] = _mm256_loadu_ps(&A.data[(5 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_5[1] = _mm256_loadu_ps(&A.data[(5 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_5[0] = yReg[0];
    var2_5[1] = yReg[1];
    var2_5[0] = _mm256_fmadd_ps(var0_5[0], var1_5[0], var2_5[0]);
    var2_5[1] = _mm256_fmadd_ps(var0_5[1], var1_5[1], var2_5[1]);
    yReg[0] = var2_5[0];
    yReg[1] = var2_5[1];
    __m256 var0_6[2];
    __m256 var1_6[2];
    __m256 var2_6[2];
    var0_6[0] = _mm256_broadcast_ss(&alphaXi[6]);
    var0_6[1] = _mm256_broadcast_ss(&alphaXi[6]);
    var1_6[0] = _mm256_loadu_ps(&A.data[(6 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_6[1] = _mm256_loadu_ps(&A.data[(6 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_6[0] = yReg[0];
    var2_6[1] = yReg[1];
    var2_6[0] = _mm256_fmadd_ps(var0_6[0], var1_6[0], var2_6[0]);
    var2_6[1] = _mm256_fmadd_ps(var0_6[1], var1_6[1], var2_6[1]);
    yReg[0] = var2_6[0];
    yReg[1] = var2_6[1];
    __m256 var0_7[2];
    __m256 var1_7[2];
    __m256 var2_7[2];
    var0_7[0] = _mm256_broadcast_ss(&alphaXi[7]);
    var0_7[1] = _mm256_broadcast_ss(&alphaXi[7]);
    var1_7[0] = _mm256_loadu_ps(&A.data[(7 + 8 * io) * (A.strides[0]) + 16 * joo]);
    var1_7[1] = _mm256_loadu_ps(&A.data[(7 + 8 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2_7[0] = yReg[0];
    var2_7[1] = yReg[1];
    var2_7[0] = _mm256_fmadd_ps(var0_7[0], var1_7[0], var2_7[0]);
    var2_7[1] = _mm256_fmadd_ps(var0_7[1], var1_7[1], var2_7[1]);
    yReg[0] = var2_7[0];
    yReg[1] = var2_7[1];
    _mm256_storeu_ps(&y.data[16 * joo], yReg[0]);
    _mm256_storeu_ps(&y.data[8 + 16 * joo], yReg[1]);
  }
  for (int_fast32_t joi = 0; joi < ((n) / (8)) % 2; joi++) {
    __m256 var0;
    var0 = _mm256_broadcast_ss(&alphaXi[0]);
    __m256 var1;
    var1 = _mm256_loadu_ps(&A.data[(8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2;
    var2 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2 = _mm256_fmadd_ps(var0, var1, var2);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2);
    __m256 var0_1;
    var0_1 = _mm256_broadcast_ss(&alphaXi[1]);
    __m256 var1_1;
    var1_1 = _mm256_loadu_ps(&A.data[(1 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_1;
    var2_1 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_1 = _mm256_fmadd_ps(var0_1, var1_1, var2_1);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_1);
    __m256 var0_2;
    var0_2 = _mm256_broadcast_ss(&alphaXi[2]);
    __m256 var1_2;
    var1_2 = _mm256_loadu_ps(&A.data[(2 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_2;
    var2_2 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_2 = _mm256_fmadd_ps(var0_2, var1_2, var2_2);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_2);
    __m256 var0_3;
    var0_3 = _mm256_broadcast_ss(&alphaXi[3]);
    __m256 var1_3;
    var1_3 = _mm256_loadu_ps(&A.data[(3 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_3;
    var2_3 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_3 = _mm256_fmadd_ps(var0_3, var1_3, var2_3);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_3);
    __m256 var0_4;
    var0_4 = _mm256_broadcast_ss(&alphaXi[4]);
    __m256 var1_4;
    var1_4 = _mm256_loadu_ps(&A.data[(4 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_4;
    var2_4 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_4 = _mm256_fmadd_ps(var0_4, var1_4, var2_4);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_4);
    __m256 var0_5;
    var0_5 = _mm256_broadcast_ss(&alphaXi[5]);
    __m256 var1_5;
    var1_5 = _mm256_loadu_ps(&A.data[(5 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_5;
    var2_5 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_5 = _mm256_fmadd_ps(var0_5, var1_5, var2_5);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_5);
    __m256 var0_6;
    var0_6 = _mm256_broadcast_ss(&alphaXi[6]);
    __m256 var1_6;
    var1_6 = _mm256_loadu_ps(&A.data[(6 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_6;
    var2_6 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_6 = _mm256_fmadd_ps(var0_6, var1_6, var2_6);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_6);
    __m256 var0_7;
    var0_7 = _mm256_broadcast_ss(&alphaXi[7]);
    __m256 var1_7;
    var1_7 = _mm256_loadu_ps(&A.data[(7 + 8 * io) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2_7;
    var2_7 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2_7 = _mm256_fmadd_ps(var0_7, var1_7, var2_7);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2_7);
  }
  for (int_fast32_t ji = 0; ji < n % 8; ji++) {
    y.data[ji + (n / 8) * 8] += alphaXi[0] * A.data[8 * io * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[1] * A.data[(1 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[2] * A.data[(2 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[3] * A.data[(3 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[4] * A.data[(4 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[5] * A.data[(5 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[6] * A.data[(6 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
    y.data[ji + (n / 8) * 8] += alphaXi[7] * A.data[(7 + 8 * io) * A.strides[0] + ji + (n / 8) * 8];
  }
}
for (int_fast32_t ii = 0; ii < m % 8; ii++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 8) * 8];
  for (int_fast32_t joo = 0; joo < ((n) / (16)); joo++) {
    __m256 var0[2];
    __m256 var1[2];
    __m256 var2[2];
    var0[0] = _mm256_broadcast_ss((&alphaXi));
    var0[1] = _mm256_broadcast_ss((&alphaXi));
    var1[0] = _mm256_loadu_ps(&A.data[(8 * (m / 8) + ii) * (A.strides[0]) + 16 * joo]);
    var1[1] = _mm256_loadu_ps(&A.data[(8 * (m / 8) + ii) * (A.strides[0]) + 8 + 16 * joo]);
    var2[0] = _mm256_loadu_ps(&y.data[16 * joo]);
    var2[1] = _mm256_loadu_ps(&y.data[8 + 16 * joo]);
    var2[0] = _mm256_fmadd_ps(var0[0], var1[0], var2[0]);
    var2[1] = _mm256_fmadd_ps(var0[1], var1[1], var2[1]);
    _mm256_storeu_ps(&y.data[16 * joo], var2[0]);
    _mm256_storeu_ps(&y.data[8 + 16 * joo], var2[1]);
  }
  for (int_fast32_t joi = 0; joi < ((n) / (8)) % 2; joi++) {
    __m256 var0;
    var0 = _mm256_broadcast_ss((&alphaXi));
    __m256 var1;
    var1 = _mm256_loadu_ps(&A.data[(8 * (m / 8) + ii) * (A.strides[0]) + 16 * (n / 16) + 8 * joi]);
    __m256 var2;
    var2 = _mm256_loadu_ps(&y.data[16 * (n / 16) + 8 * joi]);
    var2 = _mm256_fmadd_ps(var0, var1, var2);
    _mm256_storeu_ps(&y.data[16 * (n / 16) + 8 * joi], var2);
  }
  for (int_fast32_t ji = 0; ji < n % 8; ji++) {
    y.data[ji + (n / 8) * 8] += alphaXi * A.data[(ii + (m / 8) * 8) * A.strides[0] + ji + (n / 8) * 8];
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
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] = *beta * y.data[i * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  float alphaXi;
  alphaXi = *alpha * x.data[i * x.strides[0]];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j * y.strides[0]] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}


/* relying on the following instruction..."
mm256_broadcast_sd(out,val)
{out_data} = _mm256_broadcast_sd(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_sd_scalar(out,val)
{out_data} = _mm256_broadcast_sd({val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss_scalar(out,val)
{out_data} = _mm256_broadcast_ss({val_data});
*/

/* relying on the following instruction..."
mm256_fmadd_pd(dst,src1,src2)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps(dst,src1,src2)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_loadu_pd(dst,src)
{dst_data} = _mm256_loadu_pd(&{src_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_storeu_pd(dst,src)
_mm256_storeu_pd(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
