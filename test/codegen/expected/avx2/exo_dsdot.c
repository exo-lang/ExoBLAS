#include "exo_dsdot.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction..."
avx2_assoc_reduce_add_pd(x,result)

    {{
        __m256d tmp = _mm256_hadd_pd({x_data}, {x_data});
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *{result} += _mm256_cvtsd_f64(tmp);
    }}
    
*/

/* relying on the following instruction..."
avx2_convert_f32_lower_to_f64(dst,src)
{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 0));
*/

/* relying on the following instruction..."
avx2_convert_f32_upper_to_f64(dst,src)
{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 1));
*/

/* relying on the following instruction..."
avx2_reduce_add_wide_pd(dst,src)
{dst_data} = _mm256_add_pd({src_data}, {dst_data});
*/
// exo_dsdot_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dsdot_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32c y, double* result ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double d_dot;
d_dot = 0.0;
__m256d reg;
reg = _mm256_setzero_pd();
__m256d reg_1[4];
reg_1[0] = _mm256_setzero_pd();
reg_1[1] = _mm256_setzero_pd();
reg_1[2] = _mm256_setzero_pd();
reg_1[3] = _mm256_setzero_pd();
for (int_fast32_t iooo = 0; iooo < ((n) / (32)); iooo++) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256d d_x[4][2];
  __m256d d_y[4][2];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * iooo]);
  xReg[1] = _mm256_loadu_ps(&x.data[32 * iooo + 8]);
  xReg[2] = _mm256_loadu_ps(&x.data[32 * iooo + 16]);
  xReg[3] = _mm256_loadu_ps(&x.data[32 * iooo + 24]);
  yReg[0] = _mm256_loadu_ps(&y.data[32 * iooo]);
  yReg[1] = _mm256_loadu_ps(&y.data[32 * iooo + 8]);
  yReg[2] = _mm256_loadu_ps(&y.data[32 * iooo + 16]);
  yReg[3] = _mm256_loadu_ps(&y.data[32 * iooo + 24]);
  d_x[0][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[0], 0));
  d_x[1][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[1], 0));
  d_x[2][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[2], 0));
  d_x[3][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[3], 0));
  d_x[0][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[0], 1));
  d_x[1][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[1], 1));
  d_x[2][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[2], 1));
  d_x[3][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[3], 1));
  d_y[0][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[0], 0));
  d_y[1][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[1], 0));
  d_y[2][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[2], 0));
  d_y[3][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[3], 0));
  d_y[0][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[0], 1));
  d_y[1][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[1], 1));
  d_y[2][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[2], 1));
  d_y[3][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[3], 1));
  reg_1[0] = _mm256_fmadd_pd(d_x[0][0], d_y[0][0], reg_1[0]);
  reg_1[1] = _mm256_fmadd_pd(d_x[1][0], d_y[1][0], reg_1[1]);
  reg_1[2] = _mm256_fmadd_pd(d_x[2][0], d_y[2][0], reg_1[2]);
  reg_1[3] = _mm256_fmadd_pd(d_x[3][0], d_y[3][0], reg_1[3]);
  reg_1[0] = _mm256_fmadd_pd(d_x[0][1], d_y[0][1], reg_1[0]);
  reg_1[1] = _mm256_fmadd_pd(d_x[1][1], d_y[1][1], reg_1[1]);
  reg_1[2] = _mm256_fmadd_pd(d_x[2][1], d_y[2][1], reg_1[2]);
  reg_1[3] = _mm256_fmadd_pd(d_x[3][1], d_y[3][1], reg_1[3]);
}
reg = _mm256_add_pd(reg_1[0], reg);
reg = _mm256_add_pd(reg_1[1], reg);
reg = _mm256_add_pd(reg_1[2], reg);
reg = _mm256_add_pd(reg_1[3], reg);
for (int_fast32_t iooi = 0; iooi < ((n) / (8)) % 4; iooi++) {
  __m256 xReg;
  xReg = _mm256_loadu_ps(&x.data[32 * (n / 32) + 8 * iooi]);
  __m256 yReg;
  yReg = _mm256_loadu_ps(&y.data[32 * (n / 32) + 8 * iooi]);
  __m256d d_x[2];
  __m256d d_y[2];
  d_x[0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg, 0));
  d_x[1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg, 1));
  d_y[0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg, 0));
  d_y[1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg, 1));
  reg = _mm256_fmadd_pd(d_x[0], d_y[0], reg);
  reg = _mm256_fmadd_pd(d_x[1], d_y[1], reg);
}

    {
        __m256d tmp = _mm256_hadd_pd(reg, reg);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&d_dot) += _mm256_cvtsd_f64(tmp);
    }
    
for (int_fast32_t ioi = 0; ioi < ((n) / (4)) % 2; ioi++) {
  for (int_fast32_t ii = 0; ii < 4; ii++) {
    double d_x;
    d_x = (double)(x.data[4 * (ioi + (n / 8) * 2) + ii]);
    double d_y;
    d_y = (double)(y.data[4 * (ioi + (n / 8) * 2) + ii]);
    d_dot += d_x * d_y;
  }
}
for (int_fast32_t ii = 0; ii < n % 4; ii++) {
  double d_x;
  d_x = (double)(x.data[ii + (n / 4) * 4]);
  double d_y;
  d_y = (double)(y.data[ii + (n / 4) * 4]);
  d_dot += d_x * d_y;
}
*result = d_dot;
}

// exo_dsdot_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dsdot_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32c y, double* result ) {
double d_dot;
d_dot = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  double d_x;
  d_x = (double)(x.data[i * x.strides[0]]);
  double d_y;
  d_y = (double)(y.data[i * y.strides[0]]);
  d_dot += d_x * d_y;
}
*result = d_dot;
}

// exo_sdsdot_stride_1(
//     n : size,
//     sb : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sdsdot_stride_1( void *ctxt, int_fast32_t n, const float* sb, struct exo_win_1f32c x, struct exo_win_1f32c y, float* result ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double d_result;
double d_dot;
d_dot = 0.0;
__m256d reg;
reg = _mm256_setzero_pd();
__m256d reg_1[4];
reg_1[0] = _mm256_setzero_pd();
reg_1[1] = _mm256_setzero_pd();
reg_1[2] = _mm256_setzero_pd();
reg_1[3] = _mm256_setzero_pd();
for (int_fast32_t iooo = 0; iooo < ((n) / (32)); iooo++) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256d d_x[4][2];
  __m256d d_y[4][2];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * iooo]);
  xReg[1] = _mm256_loadu_ps(&x.data[32 * iooo + 8]);
  xReg[2] = _mm256_loadu_ps(&x.data[32 * iooo + 16]);
  xReg[3] = _mm256_loadu_ps(&x.data[32 * iooo + 24]);
  yReg[0] = _mm256_loadu_ps(&y.data[32 * iooo]);
  yReg[1] = _mm256_loadu_ps(&y.data[32 * iooo + 8]);
  yReg[2] = _mm256_loadu_ps(&y.data[32 * iooo + 16]);
  yReg[3] = _mm256_loadu_ps(&y.data[32 * iooo + 24]);
  d_x[0][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[0], 0));
  d_x[1][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[1], 0));
  d_x[2][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[2], 0));
  d_x[3][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[3], 0));
  d_x[0][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[0], 1));
  d_x[1][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[1], 1));
  d_x[2][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[2], 1));
  d_x[3][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg[3], 1));
  d_y[0][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[0], 0));
  d_y[1][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[1], 0));
  d_y[2][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[2], 0));
  d_y[3][0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[3], 0));
  d_y[0][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[0], 1));
  d_y[1][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[1], 1));
  d_y[2][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[2], 1));
  d_y[3][1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg[3], 1));
  reg_1[0] = _mm256_fmadd_pd(d_x[0][0], d_y[0][0], reg_1[0]);
  reg_1[1] = _mm256_fmadd_pd(d_x[1][0], d_y[1][0], reg_1[1]);
  reg_1[2] = _mm256_fmadd_pd(d_x[2][0], d_y[2][0], reg_1[2]);
  reg_1[3] = _mm256_fmadd_pd(d_x[3][0], d_y[3][0], reg_1[3]);
  reg_1[0] = _mm256_fmadd_pd(d_x[0][1], d_y[0][1], reg_1[0]);
  reg_1[1] = _mm256_fmadd_pd(d_x[1][1], d_y[1][1], reg_1[1]);
  reg_1[2] = _mm256_fmadd_pd(d_x[2][1], d_y[2][1], reg_1[2]);
  reg_1[3] = _mm256_fmadd_pd(d_x[3][1], d_y[3][1], reg_1[3]);
}
reg = _mm256_add_pd(reg_1[0], reg);
reg = _mm256_add_pd(reg_1[1], reg);
reg = _mm256_add_pd(reg_1[2], reg);
reg = _mm256_add_pd(reg_1[3], reg);
for (int_fast32_t iooi = 0; iooi < ((n) / (8)) % 4; iooi++) {
  __m256 xReg;
  xReg = _mm256_loadu_ps(&x.data[32 * (n / 32) + 8 * iooi]);
  __m256 yReg;
  yReg = _mm256_loadu_ps(&y.data[32 * (n / 32) + 8 * iooi]);
  __m256d d_x[2];
  __m256d d_y[2];
  d_x[0] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg, 0));
  d_x[1] = _mm256_cvtps_pd(_mm256_extractf128_ps(xReg, 1));
  d_y[0] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg, 0));
  d_y[1] = _mm256_cvtps_pd(_mm256_extractf128_ps(yReg, 1));
  reg = _mm256_fmadd_pd(d_x[0], d_y[0], reg);
  reg = _mm256_fmadd_pd(d_x[1], d_y[1], reg);
}

    {
        __m256d tmp = _mm256_hadd_pd(reg, reg);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&d_dot) += _mm256_cvtsd_f64(tmp);
    }
    
for (int_fast32_t ioi = 0; ioi < ((n) / (4)) % 2; ioi++) {
  for (int_fast32_t ii = 0; ii < 4; ii++) {
    double d_x;
    d_x = (double)(x.data[4 * (ioi + (n / 8) * 2) + ii]);
    double d_y;
    d_y = (double)(y.data[4 * (ioi + (n / 8) * 2) + ii]);
    d_dot += d_x * d_y;
  }
}
for (int_fast32_t ii = 0; ii < n % 4; ii++) {
  double d_x;
  d_x = (double)(x.data[ii + (n / 4) * 4]);
  double d_y;
  d_y = (double)(y.data[ii + (n / 4) * 4]);
  d_dot += d_x * d_y;
}
d_result = d_dot;
d_result += (double)(*sb);
*result = (float)(d_result);
}

// exo_sdsdot_stride_any(
//     n : size,
//     sb : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sdsdot_stride_any( void *ctxt, int_fast32_t n, const float* sb, struct exo_win_1f32c x, struct exo_win_1f32c y, float* result ) {
double d_result;
double d_dot;
d_dot = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  double d_x;
  d_x = (double)(x.data[i * x.strides[0]]);
  double d_y;
  d_y = (double)(y.data[i * y.strides[0]]);
  d_dot += d_x * d_y;
}
d_result = d_dot;
d_result += (double)(*sb);
*result = (float)(d_result);
}


/* relying on the following instruction..."
mm256_fmadd_pd(dst,src1,src2)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_setzero_pd(dst)
{dst_data} = _mm256_setzero_pd();
*/
