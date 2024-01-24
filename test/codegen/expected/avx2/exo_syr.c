#include "exo_syr.h"



static int exo_floor_div(int num, int quot) {
  int off = (num>=0)? 0 : quot-1;
  return (num-off)/quot;
}

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>



// exo_dsyr_row_major_Lower_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  __m256d var0;
  if (((1 + i) / (4)) > 0) {
    var0 = _mm256_broadcast_sd((alpha));
  }
  __m256d var1;
  if (((1 + i) / (4)) > 0) {
    var1 = _mm256_broadcast_sd(&x.data[i]);
  }
  __m256d var2;
  if (((1 + i) / (4)) > 0) {
    var2 = _mm256_mul_pd(var0, var1);
  }
  for (int_fast32_t jo = 0; jo < ((1 + i) / (4)); jo++) {
    __m256d var3;
    var3 = _mm256_loadu_pd(&x.data[4 * jo]);
    __m256d var4;
    var4 = _mm256_loadu_pd(&A.data[(i) * (A.strides[0]) + 4 * jo]);
    var4 = _mm256_fmadd_pd(var2, var3, var4);
    _mm256_storeu_pd(&A.data[(i) * (A.strides[0]) + 4 * jo], var4);
  }
  for (int_fast32_t ji = 0; ji < (1 + i) % 4; ji++) {
    A.data[i * A.strides[0] + ji + ((1 + i) / 4) * 4] += *alpha * x.data[i] * x.data[ji + ((1 + i) / 4) * 4];
  }
}
}

// exo_dsyr_row_major_Lower_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * x.data[j * x.strides[0]];
  }
}
}

// exo_dsyr_row_major_Upper_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  __m256d var0;
  if (exo_floor_div(-i + n, 4) > 0) {
    var0 = _mm256_broadcast_sd((alpha));
  }
  __m256d var1;
  if (exo_floor_div(-i + n, 4) > 0) {
    var1 = _mm256_broadcast_sd(&x.data[i]);
  }
  __m256d var2;
  if (exo_floor_div(-i + n, 4) > 0) {
    var2 = _mm256_mul_pd(var0, var1);
  }
  for (int_fast32_t jo = 0; jo < exo_floor_div(-i + n, 4); jo++) {
    __m256d var3;
    var3 = _mm256_loadu_pd(&x.data[i + 4 * jo]);
    __m256d var4;
    var4 = _mm256_loadu_pd(&A.data[(i) * (A.strides[0]) + i + 4 * jo]);
    var4 = _mm256_fmadd_pd(var2, var3, var4);
    _mm256_storeu_pd(&A.data[(i) * (A.strides[0]) + i + 4 * jo], var4);
  }
  for (int_fast32_t ji = 0; ji < (-i + n) % 4; ji++) {
    A.data[i * A.strides[0] + i + ji + exo_floor_div((-i + n), 4) * 4] += *alpha * x.data[i] * x.data[i + ji + exo_floor_div((-i + n), 4) * 4];
  }
}
}

// exo_dsyr_row_major_Upper_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     A : [f64][n, n] @DRAM
// )
void exo_dsyr_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_2f64 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < -i + n; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i * x.strides[0]] * x.data[(i + j) * x.strides[0]];
  }
}
}

// exo_ssyr_row_major_Lower_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr_row_major_Lower_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  __m256 var0;
  if (((1 + i) / (8)) > 0) {
    var0 = _mm256_broadcast_ss((alpha));
  }
  __m256 var1;
  if (((1 + i) / (8)) > 0) {
    var1 = _mm256_broadcast_ss(&x.data[i]);
  }
  __m256 var2;
  if (((1 + i) / (8)) > 0) {
    var2 = _mm256_mul_ps(var0, var1);
  }
  for (int_fast32_t jo = 0; jo < ((1 + i) / (8)); jo++) {
    __m256 var3;
    var3 = _mm256_loadu_ps(&x.data[8 * jo]);
    __m256 var4;
    var4 = _mm256_loadu_ps(&A.data[(i) * (A.strides[0]) + 8 * jo]);
    var4 = _mm256_fmadd_ps(var2, var3, var4);
    _mm256_storeu_ps(&A.data[(i) * (A.strides[0]) + 8 * jo], var4);
  }
  for (int_fast32_t ji = 0; ji < (1 + i) % 8; ji++) {
    A.data[i * A.strides[0] + ji + ((1 + i) / 8) * 8] += *alpha * x.data[i] * x.data[ji + ((1 + i) / 8) * 8];
  }
}
}

// exo_ssyr_row_major_Lower_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr_row_major_Lower_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    A.data[i * A.strides[0] + j] += *alpha * x.data[i * x.strides[0]] * x.data[j * x.strides[0]];
  }
}
}

// exo_ssyr_row_major_Upper_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr_row_major_Upper_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
for (int_fast32_t i = 0; i < n; i++) {
  __m256 var0;
  if (exo_floor_div(-i + n, 8) > 0) {
    var0 = _mm256_broadcast_ss((alpha));
  }
  __m256 var1;
  if (exo_floor_div(-i + n, 8) > 0) {
    var1 = _mm256_broadcast_ss(&x.data[i]);
  }
  __m256 var2;
  if (exo_floor_div(-i + n, 8) > 0) {
    var2 = _mm256_mul_ps(var0, var1);
  }
  for (int_fast32_t jo = 0; jo < exo_floor_div(-i + n, 8); jo++) {
    __m256 var3;
    var3 = _mm256_loadu_ps(&x.data[i + 8 * jo]);
    __m256 var4;
    var4 = _mm256_loadu_ps(&A.data[(i) * (A.strides[0]) + i + 8 * jo]);
    var4 = _mm256_fmadd_ps(var2, var3, var4);
    _mm256_storeu_ps(&A.data[(i) * (A.strides[0]) + i + 8 * jo], var4);
  }
  for (int_fast32_t ji = 0; ji < (-i + n) % 8; ji++) {
    A.data[i * A.strides[0] + i + ji + exo_floor_div((-i + n), 8) * 8] += *alpha * x.data[i] * x.data[i + ji + exo_floor_div((-i + n), 8) * 8];
  }
}
}

// exo_ssyr_row_major_Upper_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     A : [f32][n, n] @DRAM
// )
void exo_ssyr_row_major_Upper_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_2f32 A ) {
// assert stride(A, 1) == 1
for (int_fast32_t i = 0; i < n; i++) {
  for (int_fast32_t j = 0; j < -i + n; j++) {
    A.data[i * A.strides[0] + i + j] += *alpha * x.data[i * x.strides[0]] * x.data[(i + j) * x.strides[0]];
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
mm256_fmadd_reduce_pd(dst,src1,src2)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_fmadd_reduce_ps(dst,src1,src2)
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
mm256_mul_pd(out,x,y)
{out_data} = _mm256_mul_pd({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_mul_ps(out,x,y)
{out_data} = _mm256_mul_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_storeu_pd(dst,src)
_mm256_storeu_pd(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
