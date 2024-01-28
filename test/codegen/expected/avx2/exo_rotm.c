#include "exo_rotm.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction..."
avx2_prefix_sign_pd(dst,src,bound)

{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd({src_data}, _mm256_set1_pd(-1.0f));
{dst_data} = _mm256_blendv_pd ({dst_data}, src_sign, _mm256_castsi256_pd(cmp));
}}

*/

/* relying on the following instruction..."
avx2_prefix_sign_ps(dst,src,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps({src_data}, _mm256_set1_ps(-1.0f));;
{dst_data} = _mm256_blendv_ps ({dst_data}, src_sign, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
avx2_sign_pd(dst,src)
{dst_data} = _mm256_mul_pd({src_data}, _mm256_set1_pd(-1.0f));
*/

/* relying on the following instruction..."
avx2_sign_ps(dst,src)
{dst_data} = _mm256_mul_ps({src_data}, _mm256_set1_ps(-1.0f));
*/
// exo_drotm_flag_neg_one_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_neg_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256d var3;
var3 = _mm256_broadcast_sd(&H[1]);
__m256d var1;
var1 = _mm256_broadcast_sd(&H[0]);
__m256d var7;
var7 = _mm256_broadcast_sd(&H[2 + 1]);
__m256d var5;
var5 = _mm256_broadcast_sd(&H[2]);
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var2[4];
  __m256d var4[4];
  __m256d var6[4];
  xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  var2[0] = _mm256_mul_pd(var3, yReg[0]);
  var2[1] = _mm256_mul_pd(var3, yReg[1]);
  var2[2] = _mm256_mul_pd(var3, yReg[2]);
  var2[3] = _mm256_mul_pd(var3, yReg[3]);
  var0[0] = _mm256_fmadd_pd(var1, xReg[0], var2[0]);
  var0[1] = _mm256_fmadd_pd(var1, xReg[1], var2[1]);
  var0[2] = _mm256_fmadd_pd(var1, xReg[2], var2[2]);
  var0[3] = _mm256_fmadd_pd(var1, xReg[3], var2[3]);
  _mm256_storeu_pd(&x.data[16 * ioo], var0[0]);
  _mm256_storeu_pd(&x.data[4 + 16 * ioo], var0[1]);
  _mm256_storeu_pd(&x.data[8 + 16 * ioo], var0[2]);
  _mm256_storeu_pd(&x.data[12 + 16 * ioo], var0[3]);
  var6[0] = _mm256_mul_pd(var7, yReg[0]);
  var6[1] = _mm256_mul_pd(var7, yReg[1]);
  var6[2] = _mm256_mul_pd(var7, yReg[2]);
  var6[3] = _mm256_mul_pd(var7, yReg[3]);
  var4[0] = _mm256_fmadd_pd(var5, xReg[0], var6[0]);
  var4[1] = _mm256_fmadd_pd(var5, xReg[1], var6[1]);
  var4[2] = _mm256_fmadd_pd(var5, xReg[2], var6[2]);
  var4[3] = _mm256_fmadd_pd(var5, xReg[3], var6[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], var4[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], var4[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], var4[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], var4[3]);
}
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  __m256d var4;
  __m256d var6;
  xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  var2 = _mm256_mul_pd(var3, yReg);
  var0 = _mm256_fmadd_pd(var1, xReg, var2);
  _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var0);
  var6 = _mm256_mul_pd(var7, yReg);
  var4 = _mm256_fmadd_pd(var5, xReg, var6);
  _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var4);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  __m256d var4;
  __m256d var6;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var3, yReg);
var2 = _mm256_blendv_pd (var2, mul, _mm256_castsi256_pd(cmp));
}

  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var1, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, xReg, var2);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var0);
    }
    
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var7, yReg);
var6 = _mm256_blendv_pd (var6, mul, _mm256_castsi256_pd(cmp));
}

  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var5, _mm256_castsi256_pd(cmp));
    var4 = _mm256_fmadd_pd(prefixed_src1, xReg, var6);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var4);
    }
    
}
}

// exo_drotm_flag_neg_one_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_neg_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = H[0] * xReg + H[1] * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = H[2] * xReg + H[3] * y.data[i * y.strides[0]];
}
}

// exo_drotm_flag_one_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256d var1;
var1 = _mm256_broadcast_sd(&H[0]);
__m256d var4;
var4 = _mm256_broadcast_sd(&H[2 + 1]);
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var2[4];
  __m256d var3[4];
  xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  var0[0] = _mm256_fmadd_pd(var1, xReg[0], yReg[0]);
  var0[1] = _mm256_fmadd_pd(var1, xReg[1], yReg[1]);
  var0[2] = _mm256_fmadd_pd(var1, xReg[2], yReg[2]);
  var0[3] = _mm256_fmadd_pd(var1, xReg[3], yReg[3]);
  _mm256_storeu_pd(&x.data[16 * ioo], var0[0]);
  _mm256_storeu_pd(&x.data[4 + 16 * ioo], var0[1]);
  _mm256_storeu_pd(&x.data[8 + 16 * ioo], var0[2]);
  _mm256_storeu_pd(&x.data[12 + 16 * ioo], var0[3]);
  var3[0] = _mm256_mul_pd(xReg[0], _mm256_set1_pd(-1.0f));
  var3[1] = _mm256_mul_pd(xReg[1], _mm256_set1_pd(-1.0f));
  var3[2] = _mm256_mul_pd(xReg[2], _mm256_set1_pd(-1.0f));
  var3[3] = _mm256_mul_pd(xReg[3], _mm256_set1_pd(-1.0f));
  var2[0] = _mm256_fmadd_pd(var4, yReg[0], var3[0]);
  var2[1] = _mm256_fmadd_pd(var4, yReg[1], var3[1]);
  var2[2] = _mm256_fmadd_pd(var4, yReg[2], var3[2]);
  var2[3] = _mm256_fmadd_pd(var4, yReg[3], var3[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], var2[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], var2[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], var2[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], var2[3]);
}
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  __m256d var3;
  xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  var0 = _mm256_fmadd_pd(var1, xReg, yReg);
  _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var0);
  var3 = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
  var2 = _mm256_fmadd_pd(var4, yReg, var3);
  _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var2);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  __m256d var3;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var1, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, xReg, yReg);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var0);
    }
    
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
var3 = _mm256_blendv_pd (var3, src_sign, _mm256_castsi256_pd(cmp));
}

  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var4, _mm256_castsi256_pd(cmp));
    var2 = _mm256_fmadd_pd(prefixed_src1, yReg, var3);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var2);
    }
    
}
}

// exo_drotm_flag_one_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = H[0] * xReg + y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = -xReg + H[3] * y.data[i * y.strides[0]];
}
}

// exo_drotm_flag_zero_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_zero_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256d var1;
var1 = _mm256_broadcast_sd(&H[1]);
__m256d var3;
var3 = _mm256_broadcast_sd(&H[2]);
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var2[4];
  xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  var0[0] = _mm256_fmadd_pd(var1, yReg[0], xReg[0]);
  var0[1] = _mm256_fmadd_pd(var1, yReg[1], xReg[1]);
  var0[2] = _mm256_fmadd_pd(var1, yReg[2], xReg[2]);
  var0[3] = _mm256_fmadd_pd(var1, yReg[3], xReg[3]);
  _mm256_storeu_pd(&x.data[16 * ioo], var0[0]);
  _mm256_storeu_pd(&x.data[4 + 16 * ioo], var0[1]);
  _mm256_storeu_pd(&x.data[8 + 16 * ioo], var0[2]);
  _mm256_storeu_pd(&x.data[12 + 16 * ioo], var0[3]);
  var2[0] = _mm256_fmadd_pd(var3, xReg[0], yReg[0]);
  var2[1] = _mm256_fmadd_pd(var3, xReg[1], yReg[1]);
  var2[2] = _mm256_fmadd_pd(var3, xReg[2], yReg[2]);
  var2[3] = _mm256_fmadd_pd(var3, xReg[3], yReg[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], var2[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], var2[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], var2[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], var2[3]);
}
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  var0 = _mm256_fmadd_pd(var1, yReg, xReg);
  _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var0);
  var2 = _mm256_fmadd_pd(var3, xReg, yReg);
  _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var2);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  __m256d var2;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var1, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, yReg, xReg);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var0);
    }
    
  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var3, _mm256_castsi256_pd(cmp));
    var2 = _mm256_fmadd_pd(prefixed_src1, xReg, yReg);
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var2);
    }
    
}
}

// exo_drotm_flag_zero_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     H : f64[2, 2] @DRAM
// )
void exo_drotm_flag_zero_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = xReg + H[1] * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = H[2] * xReg + y.data[i * y.strides[0]];
}
}

// exo_srotm_flag_neg_one_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_neg_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256 var3;
var3 = _mm256_broadcast_ss(&H[1]);
__m256 var1;
var1 = _mm256_broadcast_ss(&H[0]);
__m256 var7;
var7 = _mm256_broadcast_ss(&H[2 + 1]);
__m256 var5;
var5 = _mm256_broadcast_ss(&H[2]);
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var2[4];
  __m256 var4[4];
  __m256 var6[4];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  var2[0] = _mm256_mul_ps(var3, yReg[0]);
  var2[1] = _mm256_mul_ps(var3, yReg[1]);
  var2[2] = _mm256_mul_ps(var3, yReg[2]);
  var2[3] = _mm256_mul_ps(var3, yReg[3]);
  var0[0] = _mm256_fmadd_ps(var1, xReg[0], var2[0]);
  var0[1] = _mm256_fmadd_ps(var1, xReg[1], var2[1]);
  var0[2] = _mm256_fmadd_ps(var1, xReg[2], var2[2]);
  var0[3] = _mm256_fmadd_ps(var1, xReg[3], var2[3]);
  _mm256_storeu_ps(&x.data[32 * ioo], var0[0]);
  _mm256_storeu_ps(&x.data[8 + 32 * ioo], var0[1]);
  _mm256_storeu_ps(&x.data[16 + 32 * ioo], var0[2]);
  _mm256_storeu_ps(&x.data[24 + 32 * ioo], var0[3]);
  var6[0] = _mm256_mul_ps(var7, yReg[0]);
  var6[1] = _mm256_mul_ps(var7, yReg[1]);
  var6[2] = _mm256_mul_ps(var7, yReg[2]);
  var6[3] = _mm256_mul_ps(var7, yReg[3]);
  var4[0] = _mm256_fmadd_ps(var5, xReg[0], var6[0]);
  var4[1] = _mm256_fmadd_ps(var5, xReg[1], var6[1]);
  var4[2] = _mm256_fmadd_ps(var5, xReg[2], var6[2]);
  var4[3] = _mm256_fmadd_ps(var5, xReg[3], var6[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], var4[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], var4[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], var4[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], var4[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  __m256 var4;
  __m256 var6;
  xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  var2 = _mm256_mul_ps(var3, yReg);
  var0 = _mm256_fmadd_ps(var1, xReg, var2);
  _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var0);
  var6 = _mm256_mul_ps(var7, yReg);
  var4 = _mm256_fmadd_ps(var5, xReg, var6);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var4);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  __m256 var4;
  __m256 var6;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var3, yReg);
var2 = _mm256_blendv_ps (var2, mul, _mm256_castsi256_ps(cmp));
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var1, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, xReg, var2);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var0);
    }
    
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var7, yReg);
var6 = _mm256_blendv_ps (var6, mul, _mm256_castsi256_ps(cmp));
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var5, _mm256_castsi256_ps(cmp));
    var4 = _mm256_fmadd_ps(prefixed_src1, xReg, var6);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var4);
    }
    
}
}

// exo_srotm_flag_neg_one_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_neg_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = H[0] * xReg + H[1] * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = H[2] * xReg + H[3] * y.data[i * y.strides[0]];
}
}

// exo_srotm_flag_one_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_one_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256 var1;
var1 = _mm256_broadcast_ss(&H[0]);
__m256 var4;
var4 = _mm256_broadcast_ss(&H[2 + 1]);
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var2[4];
  __m256 var3[4];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  var0[0] = _mm256_fmadd_ps(var1, xReg[0], yReg[0]);
  var0[1] = _mm256_fmadd_ps(var1, xReg[1], yReg[1]);
  var0[2] = _mm256_fmadd_ps(var1, xReg[2], yReg[2]);
  var0[3] = _mm256_fmadd_ps(var1, xReg[3], yReg[3]);
  _mm256_storeu_ps(&x.data[32 * ioo], var0[0]);
  _mm256_storeu_ps(&x.data[8 + 32 * ioo], var0[1]);
  _mm256_storeu_ps(&x.data[16 + 32 * ioo], var0[2]);
  _mm256_storeu_ps(&x.data[24 + 32 * ioo], var0[3]);
  var3[0] = _mm256_mul_ps(xReg[0], _mm256_set1_ps(-1.0f));
  var3[1] = _mm256_mul_ps(xReg[1], _mm256_set1_ps(-1.0f));
  var3[2] = _mm256_mul_ps(xReg[2], _mm256_set1_ps(-1.0f));
  var3[3] = _mm256_mul_ps(xReg[3], _mm256_set1_ps(-1.0f));
  var2[0] = _mm256_fmadd_ps(var4, yReg[0], var3[0]);
  var2[1] = _mm256_fmadd_ps(var4, yReg[1], var3[1]);
  var2[2] = _mm256_fmadd_ps(var4, yReg[2], var3[2]);
  var2[3] = _mm256_fmadd_ps(var4, yReg[3], var3[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], var2[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], var2[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], var2[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], var2[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  __m256 var3;
  xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  var0 = _mm256_fmadd_ps(var1, xReg, yReg);
  _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var0);
  var3 = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));
  var2 = _mm256_fmadd_ps(var4, yReg, var3);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var2);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  __m256 var3;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var1, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, xReg, yReg);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var0);
    }
    
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));;
var3 = _mm256_blendv_ps (var3, src_sign, _mm256_castsi256_ps(cmp));
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var4, _mm256_castsi256_ps(cmp));
    var2 = _mm256_fmadd_ps(prefixed_src1, yReg, var3);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var2);
    }
    
}
}

// exo_srotm_flag_one_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_one_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = H[0] * xReg + y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = -xReg + H[3] * y.data[i * y.strides[0]];
}
}

// exo_srotm_flag_zero_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_zero_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
// assert stride(H, 1) == 1
__m256 var1;
var1 = _mm256_broadcast_ss(&H[1]);
__m256 var3;
var3 = _mm256_broadcast_ss(&H[2]);
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var2[4];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  var0[0] = _mm256_fmadd_ps(var1, yReg[0], xReg[0]);
  var0[1] = _mm256_fmadd_ps(var1, yReg[1], xReg[1]);
  var0[2] = _mm256_fmadd_ps(var1, yReg[2], xReg[2]);
  var0[3] = _mm256_fmadd_ps(var1, yReg[3], xReg[3]);
  _mm256_storeu_ps(&x.data[32 * ioo], var0[0]);
  _mm256_storeu_ps(&x.data[8 + 32 * ioo], var0[1]);
  _mm256_storeu_ps(&x.data[16 + 32 * ioo], var0[2]);
  _mm256_storeu_ps(&x.data[24 + 32 * ioo], var0[3]);
  var2[0] = _mm256_fmadd_ps(var3, xReg[0], yReg[0]);
  var2[1] = _mm256_fmadd_ps(var3, xReg[1], yReg[1]);
  var2[2] = _mm256_fmadd_ps(var3, xReg[2], yReg[2]);
  var2[3] = _mm256_fmadd_ps(var3, xReg[3], yReg[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], var2[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], var2[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], var2[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], var2[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  var0 = _mm256_fmadd_ps(var1, yReg, xReg);
  _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var0);
  var2 = _mm256_fmadd_ps(var3, xReg, yReg);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var2);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  __m256 var2;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var1, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, yReg, xReg);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var0);
    }
    
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var3, _mm256_castsi256_ps(cmp));
    var2 = _mm256_fmadd_ps(prefixed_src1, xReg, yReg);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var2);
    }
    
}
}

// exo_srotm_flag_zero_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     H : f32[2, 2] @DRAM
// )
void exo_srotm_flag_zero_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* H ) {
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = xReg + H[1] * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = H[2] * xReg + y.data[i * y.strides[0]];
}
}


/* relying on the following instruction..."
mm256_broadcast_sd(out,val)
{out_data} = _mm256_broadcast_sd(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_fmadd_pd(dst,src1,src2,src3)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
*/

/* relying on the following instruction..."
mm256_fmadd_pd_commute(dst,src1,src2,src3)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps(dst,src1,src2,src3)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps_commute(dst,src1,src2,src3)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
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
mm256_prefix_fmadd_pd(dst,src1,src2,src3,bound)

{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {src3_data});
}}

*/

/* relying on the following instruction..."
mm256_prefix_fmadd_pd_commute(dst,src1,src2,src3,bound)

{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {src3_data});
}}

*/

/* relying on the following instruction..."
mm256_prefix_fmadd_ps(dst,src1,src2,src3,bound)

{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), {src1_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_fmadd_ps(prefixed_src1, {src2_data}, {src3_data});
}}

*/

/* relying on the following instruction..."
mm256_prefix_fmadd_ps_commute(dst,src1,src2,src3,bound)

{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), {src1_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_fmadd_ps(prefixed_src1, {src2_data}, {src3_data});
}}

*/

/* relying on the following instruction..."
mm256_prefix_load_pd(dst,src,bound)

       {{
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x({bound});
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            {dst_data} = _mm256_maskload_pd(&{src_data}, cmp);
       }}
       
*/

/* relying on the following instruction..."
mm256_prefix_load_ps(dst,src,bound)

{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_maskload_ps(&{src_data}, cmp);
}}

*/

/* relying on the following instruction..."
mm256_prefix_mul_pd(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd({x_data}, {y_data});
{out_data} = _mm256_blendv_pd ({out_data}, mul, _mm256_castsi256_pd(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_mul_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, mul, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_store_pd(dst,src,bound)

    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&{dst_data}, cmp, {src_data});
    }}
    
*/

/* relying on the following instruction..."
mm256_prefix_store_ps(dst,src,bound)

    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&{dst_data}, cmp, {src_data});
    }}
    
*/

/* relying on the following instruction..."
mm256_storeu_pd(dst,src)
_mm256_storeu_pd(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
