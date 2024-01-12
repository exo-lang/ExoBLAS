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
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var1[4];
  __m256d var2[4];
  __m256d var3[4];
  __m256d var4[4];
  __m256d var5[4];
  __m256d var6[4];
  __m256d var7[4];
  __m256d var8[4];
  __m256d var9[4];
  var0[0] = _mm256_broadcast_sd(&H[0]);
  var0[1] = _mm256_broadcast_sd(&H[0]);
  var0[2] = _mm256_broadcast_sd(&H[0]);
  var0[3] = _mm256_broadcast_sd(&H[0]);
  var2[0] = _mm256_broadcast_sd(&H[1]);
  var2[1] = _mm256_broadcast_sd(&H[1]);
  var2[2] = _mm256_broadcast_sd(&H[1]);
  var2[3] = _mm256_broadcast_sd(&H[1]);
  var5[0] = _mm256_broadcast_sd(&H[2]);
  var5[1] = _mm256_broadcast_sd(&H[2]);
  var5[2] = _mm256_broadcast_sd(&H[2]);
  var5[3] = _mm256_broadcast_sd(&H[2]);
  var7[0] = _mm256_broadcast_sd(&H[2 + 1]);
  var7[1] = _mm256_broadcast_sd(&H[2 + 1]);
  var7[2] = _mm256_broadcast_sd(&H[2 + 1]);
  var7[3] = _mm256_broadcast_sd(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
    xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
    xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
    yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
    yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
    var1[0] = _mm256_mul_pd(var0[0], xReg[0]);
    var1[1] = _mm256_mul_pd(var0[1], xReg[1]);
    var1[2] = _mm256_mul_pd(var0[2], xReg[2]);
    var1[3] = _mm256_mul_pd(var0[3], xReg[3]);
    var3[0] = _mm256_mul_pd(var2[0], yReg[0]);
    var3[1] = _mm256_mul_pd(var2[1], yReg[1]);
    var3[2] = _mm256_mul_pd(var2[2], yReg[2]);
    var3[3] = _mm256_mul_pd(var2[3], yReg[3]);
    var4[0] = _mm256_add_pd(var1[0], var3[0]);
    var4[1] = _mm256_add_pd(var1[1], var3[1]);
    var4[2] = _mm256_add_pd(var1[2], var3[2]);
    var4[3] = _mm256_add_pd(var1[3], var3[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], var4[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], var4[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], var4[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], var4[3]);
    var6[0] = _mm256_mul_pd(var5[0], xReg[0]);
    var6[1] = _mm256_mul_pd(var5[1], xReg[1]);
    var6[2] = _mm256_mul_pd(var5[2], xReg[2]);
    var6[3] = _mm256_mul_pd(var5[3], xReg[3]);
    var8[0] = _mm256_mul_pd(var7[0], yReg[0]);
    var8[1] = _mm256_mul_pd(var7[1], yReg[1]);
    var8[2] = _mm256_mul_pd(var7[2], yReg[2]);
    var8[3] = _mm256_mul_pd(var7[3], yReg[3]);
    var9[0] = _mm256_add_pd(var6[0], var8[0]);
    var9[1] = _mm256_add_pd(var6[1], var8[1]);
    var9[2] = _mm256_add_pd(var6[2], var8[2]);
    var9[3] = _mm256_add_pd(var6[3], var8[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], var9[0]);
    _mm256_storeu_pd(&y.data[4 + 16 * ioo], var9[1]);
    _mm256_storeu_pd(&y.data[8 + 16 * ioo], var9[2]);
    _mm256_storeu_pd(&y.data[12 + 16 * ioo], var9[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  var0 = _mm256_broadcast_sd(&H[0]);
  __m256d var1;
  __m256d var2;
  var2 = _mm256_broadcast_sd(&H[1]);
  __m256d var3;
  __m256d var4;
  __m256d var5;
  var5 = _mm256_broadcast_sd(&H[2]);
  __m256d var6;
  __m256d var7;
  var7 = _mm256_broadcast_sd(&H[2 + 1]);
  __m256d var8;
  __m256d var9;
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    var1 = _mm256_mul_pd(var0, xReg);
    var3 = _mm256_mul_pd(var2, yReg);
    var4 = _mm256_add_pd(var1, var3);
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var4);
    var6 = _mm256_mul_pd(var5, xReg);
    var8 = _mm256_mul_pd(var7, yReg);
    var9 = _mm256_add_pd(var6, var8);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var9);
  }
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d var0;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[0]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var0, xReg);
var1 = _mm256_blendv_pd (var1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var2;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var2 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var3;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var2, yReg);
var3 = _mm256_blendv_pd (var3, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var4;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(var1, var3);
var4 = _mm256_blendv_pd (var4, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var4);
    }
    
  __m256d var5;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var5 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var6;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var5, xReg);
var6 = _mm256_blendv_pd (var6, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var7;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var7 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2 + 1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var8;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var7, yReg);
var8 = _mm256_blendv_pd (var8, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var9;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(var6, var8);
var9 = _mm256_blendv_pd (var9, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var9);
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
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var1[4];
  __m256d var2[4];
  __m256d var3[4];
  __m256d var4[4];
  __m256d var5[4];
  __m256d var6[4];
  var0[0] = _mm256_broadcast_sd(&H[0]);
  var0[1] = _mm256_broadcast_sd(&H[0]);
  var0[2] = _mm256_broadcast_sd(&H[0]);
  var0[3] = _mm256_broadcast_sd(&H[0]);
  var4[0] = _mm256_broadcast_sd(&H[2 + 1]);
  var4[1] = _mm256_broadcast_sd(&H[2 + 1]);
  var4[2] = _mm256_broadcast_sd(&H[2 + 1]);
  var4[3] = _mm256_broadcast_sd(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
    xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
    xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
    yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
    yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
    var1[0] = _mm256_mul_pd(var0[0], xReg[0]);
    var1[1] = _mm256_mul_pd(var0[1], xReg[1]);
    var1[2] = _mm256_mul_pd(var0[2], xReg[2]);
    var1[3] = _mm256_mul_pd(var0[3], xReg[3]);
    var2[0] = _mm256_add_pd(var1[0], yReg[0]);
    var2[1] = _mm256_add_pd(var1[1], yReg[1]);
    var2[2] = _mm256_add_pd(var1[2], yReg[2]);
    var2[3] = _mm256_add_pd(var1[3], yReg[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], var2[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], var2[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], var2[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], var2[3]);
    var3[0] = _mm256_mul_pd(xReg[0], _mm256_set1_pd(-1.0f));
    var3[1] = _mm256_mul_pd(xReg[1], _mm256_set1_pd(-1.0f));
    var3[2] = _mm256_mul_pd(xReg[2], _mm256_set1_pd(-1.0f));
    var3[3] = _mm256_mul_pd(xReg[3], _mm256_set1_pd(-1.0f));
    var5[0] = _mm256_mul_pd(var4[0], yReg[0]);
    var5[1] = _mm256_mul_pd(var4[1], yReg[1]);
    var5[2] = _mm256_mul_pd(var4[2], yReg[2]);
    var5[3] = _mm256_mul_pd(var4[3], yReg[3]);
    var6[0] = _mm256_add_pd(var3[0], var5[0]);
    var6[1] = _mm256_add_pd(var3[1], var5[1]);
    var6[2] = _mm256_add_pd(var3[2], var5[2]);
    var6[3] = _mm256_add_pd(var3[3], var5[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], var6[0]);
    _mm256_storeu_pd(&y.data[4 + 16 * ioo], var6[1]);
    _mm256_storeu_pd(&y.data[8 + 16 * ioo], var6[2]);
    _mm256_storeu_pd(&y.data[12 + 16 * ioo], var6[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  var0 = _mm256_broadcast_sd(&H[0]);
  __m256d var1;
  __m256d var2;
  __m256d var3;
  __m256d var4;
  var4 = _mm256_broadcast_sd(&H[2 + 1]);
  __m256d var5;
  __m256d var6;
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    var1 = _mm256_mul_pd(var0, xReg);
    var2 = _mm256_add_pd(var1, yReg);
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var2);
    var3 = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
    var5 = _mm256_mul_pd(var4, yReg);
    var6 = _mm256_add_pd(var3, var5);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var6);
  }
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d var0;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[0]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var0, xReg);
var1 = _mm256_blendv_pd (var1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(var1, yReg);
var2 = _mm256_blendv_pd (var2, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var2);
    }
    
  __m256d var3;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
var3 = _mm256_blendv_pd (var3, src_sign, _mm256_castsi256_pd(cmp));
}

  __m256d var4;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var4 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2 + 1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var5;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var4, yReg);
var5 = _mm256_blendv_pd (var5, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var6;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(var3, var5);
var6 = _mm256_blendv_pd (var6, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var6);
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
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d var0[4];
  __m256d var1[4];
  __m256d var2[4];
  __m256d var3[4];
  __m256d var4[4];
  __m256d var5[4];
  var0[0] = _mm256_broadcast_sd(&H[1]);
  var0[1] = _mm256_broadcast_sd(&H[1]);
  var0[2] = _mm256_broadcast_sd(&H[1]);
  var0[3] = _mm256_broadcast_sd(&H[1]);
  var3[0] = _mm256_broadcast_sd(&H[2]);
  var3[1] = _mm256_broadcast_sd(&H[2]);
  var3[2] = _mm256_broadcast_sd(&H[2]);
  var3[3] = _mm256_broadcast_sd(&H[2]);
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
    xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
    xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
    yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
    yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
    var1[0] = _mm256_mul_pd(var0[0], yReg[0]);
    var1[1] = _mm256_mul_pd(var0[1], yReg[1]);
    var1[2] = _mm256_mul_pd(var0[2], yReg[2]);
    var1[3] = _mm256_mul_pd(var0[3], yReg[3]);
    var2[0] = _mm256_add_pd(xReg[0], var1[0]);
    var2[1] = _mm256_add_pd(xReg[1], var1[1]);
    var2[2] = _mm256_add_pd(xReg[2], var1[2]);
    var2[3] = _mm256_add_pd(xReg[3], var1[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], var2[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], var2[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], var2[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], var2[3]);
    var4[0] = _mm256_mul_pd(var3[0], xReg[0]);
    var4[1] = _mm256_mul_pd(var3[1], xReg[1]);
    var4[2] = _mm256_mul_pd(var3[2], xReg[2]);
    var4[3] = _mm256_mul_pd(var3[3], xReg[3]);
    var5[0] = _mm256_add_pd(var4[0], yReg[0]);
    var5[1] = _mm256_add_pd(var4[1], yReg[1]);
    var5[2] = _mm256_add_pd(var4[2], yReg[2]);
    var5[3] = _mm256_add_pd(var4[3], yReg[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], var5[0]);
    _mm256_storeu_pd(&y.data[4 + 16 * ioo], var5[1]);
    _mm256_storeu_pd(&y.data[8 + 16 * ioo], var5[2]);
    _mm256_storeu_pd(&y.data[12 + 16 * ioo], var5[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d var0;
  var0 = _mm256_broadcast_sd(&H[1]);
  __m256d var1;
  __m256d var2;
  __m256d var3;
  var3 = _mm256_broadcast_sd(&H[2]);
  __m256d var4;
  __m256d var5;
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    var1 = _mm256_mul_pd(var0, yReg);
    var2 = _mm256_add_pd(xReg, var1);
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var2);
    var4 = _mm256_mul_pd(var3, xReg);
    var5 = _mm256_add_pd(var4, yReg);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], var5);
  }
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d var0;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var0, yReg);
var1 = _mm256_blendv_pd (var1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(xReg, var1);
var2 = _mm256_blendv_pd (var2, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, var2);
    }
    
  __m256d var3;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var3 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d var4;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(var3, xReg);
var4 = _mm256_blendv_pd (var4, mul, _mm256_castsi256_pd(cmp));
}

  __m256d var5;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(var4, yReg);
var5 = _mm256_blendv_pd (var5, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, var5);
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
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var1[4];
  __m256 var2[4];
  __m256 var3[4];
  __m256 var4[4];
  __m256 var5[4];
  __m256 var6[4];
  __m256 var7[4];
  __m256 var8[4];
  __m256 var9[4];
  var0[0] = _mm256_broadcast_ss(&H[0]);
  var0[1] = _mm256_broadcast_ss(&H[0]);
  var0[2] = _mm256_broadcast_ss(&H[0]);
  var0[3] = _mm256_broadcast_ss(&H[0]);
  var2[0] = _mm256_broadcast_ss(&H[1]);
  var2[1] = _mm256_broadcast_ss(&H[1]);
  var2[2] = _mm256_broadcast_ss(&H[1]);
  var2[3] = _mm256_broadcast_ss(&H[1]);
  var5[0] = _mm256_broadcast_ss(&H[2]);
  var5[1] = _mm256_broadcast_ss(&H[2]);
  var5[2] = _mm256_broadcast_ss(&H[2]);
  var5[3] = _mm256_broadcast_ss(&H[2]);
  var7[0] = _mm256_broadcast_ss(&H[2 + 1]);
  var7[1] = _mm256_broadcast_ss(&H[2 + 1]);
  var7[2] = _mm256_broadcast_ss(&H[2 + 1]);
  var7[3] = _mm256_broadcast_ss(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
    xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
    xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
    yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
    yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
    var1[0] = _mm256_mul_ps(var0[0], xReg[0]);
    var1[1] = _mm256_mul_ps(var0[1], xReg[1]);
    var1[2] = _mm256_mul_ps(var0[2], xReg[2]);
    var1[3] = _mm256_mul_ps(var0[3], xReg[3]);
    var3[0] = _mm256_mul_ps(var2[0], yReg[0]);
    var3[1] = _mm256_mul_ps(var2[1], yReg[1]);
    var3[2] = _mm256_mul_ps(var2[2], yReg[2]);
    var3[3] = _mm256_mul_ps(var2[3], yReg[3]);
    var4[0] = _mm256_add_ps(var1[0], var3[0]);
    var4[1] = _mm256_add_ps(var1[1], var3[1]);
    var4[2] = _mm256_add_ps(var1[2], var3[2]);
    var4[3] = _mm256_add_ps(var1[3], var3[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], var4[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], var4[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], var4[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], var4[3]);
    var6[0] = _mm256_mul_ps(var5[0], xReg[0]);
    var6[1] = _mm256_mul_ps(var5[1], xReg[1]);
    var6[2] = _mm256_mul_ps(var5[2], xReg[2]);
    var6[3] = _mm256_mul_ps(var5[3], xReg[3]);
    var8[0] = _mm256_mul_ps(var7[0], yReg[0]);
    var8[1] = _mm256_mul_ps(var7[1], yReg[1]);
    var8[2] = _mm256_mul_ps(var7[2], yReg[2]);
    var8[3] = _mm256_mul_ps(var7[3], yReg[3]);
    var9[0] = _mm256_add_ps(var6[0], var8[0]);
    var9[1] = _mm256_add_ps(var6[1], var8[1]);
    var9[2] = _mm256_add_ps(var6[2], var8[2]);
    var9[3] = _mm256_add_ps(var6[3], var8[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], var9[0]);
    _mm256_storeu_ps(&y.data[8 + 32 * ioo], var9[1]);
    _mm256_storeu_ps(&y.data[16 + 32 * ioo], var9[2]);
    _mm256_storeu_ps(&y.data[24 + 32 * ioo], var9[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  var0 = _mm256_broadcast_ss(&H[0]);
  __m256 var1;
  __m256 var2;
  var2 = _mm256_broadcast_ss(&H[1]);
  __m256 var3;
  __m256 var4;
  __m256 var5;
  var5 = _mm256_broadcast_ss(&H[2]);
  __m256 var6;
  __m256 var7;
  var7 = _mm256_broadcast_ss(&H[2 + 1]);
  __m256 var8;
  __m256 var9;
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    var1 = _mm256_mul_ps(var0, xReg);
    var3 = _mm256_mul_ps(var2, yReg);
    var4 = _mm256_add_ps(var1, var3);
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var4);
    var6 = _mm256_mul_ps(var5, xReg);
    var8 = _mm256_mul_ps(var7, yReg);
    var9 = _mm256_add_ps(var6, var8);
    _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var9);
  }
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 var0;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[0]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var0, xReg);
var1 = _mm256_blendv_ps (var1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var2;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var2 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var3;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var2, yReg);
var3 = _mm256_blendv_ps (var3, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var4;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var1, var3);
var4 = _mm256_blendv_ps (var4, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var4);
    }
    
  __m256 var5;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var5 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var6;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var5, xReg);
var6 = _mm256_blendv_ps (var6, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var7;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var7 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2 + 1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var8;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var7, yReg);
var8 = _mm256_blendv_ps (var8, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var9;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var6, var8);
var9 = _mm256_blendv_ps (var9, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var9);
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
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var1[4];
  __m256 var2[4];
  __m256 var3[4];
  __m256 var4[4];
  __m256 var5[4];
  __m256 var6[4];
  var0[0] = _mm256_broadcast_ss(&H[0]);
  var0[1] = _mm256_broadcast_ss(&H[0]);
  var0[2] = _mm256_broadcast_ss(&H[0]);
  var0[3] = _mm256_broadcast_ss(&H[0]);
  var4[0] = _mm256_broadcast_ss(&H[2 + 1]);
  var4[1] = _mm256_broadcast_ss(&H[2 + 1]);
  var4[2] = _mm256_broadcast_ss(&H[2 + 1]);
  var4[3] = _mm256_broadcast_ss(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
    xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
    xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
    yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
    yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
    var1[0] = _mm256_mul_ps(var0[0], xReg[0]);
    var1[1] = _mm256_mul_ps(var0[1], xReg[1]);
    var1[2] = _mm256_mul_ps(var0[2], xReg[2]);
    var1[3] = _mm256_mul_ps(var0[3], xReg[3]);
    var2[0] = _mm256_add_ps(var1[0], yReg[0]);
    var2[1] = _mm256_add_ps(var1[1], yReg[1]);
    var2[2] = _mm256_add_ps(var1[2], yReg[2]);
    var2[3] = _mm256_add_ps(var1[3], yReg[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], var2[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], var2[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], var2[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], var2[3]);
    var3[0] = _mm256_mul_ps(xReg[0], _mm256_set1_ps(-1.0f));
    var3[1] = _mm256_mul_ps(xReg[1], _mm256_set1_ps(-1.0f));
    var3[2] = _mm256_mul_ps(xReg[2], _mm256_set1_ps(-1.0f));
    var3[3] = _mm256_mul_ps(xReg[3], _mm256_set1_ps(-1.0f));
    var5[0] = _mm256_mul_ps(var4[0], yReg[0]);
    var5[1] = _mm256_mul_ps(var4[1], yReg[1]);
    var5[2] = _mm256_mul_ps(var4[2], yReg[2]);
    var5[3] = _mm256_mul_ps(var4[3], yReg[3]);
    var6[0] = _mm256_add_ps(var3[0], var5[0]);
    var6[1] = _mm256_add_ps(var3[1], var5[1]);
    var6[2] = _mm256_add_ps(var3[2], var5[2]);
    var6[3] = _mm256_add_ps(var3[3], var5[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], var6[0]);
    _mm256_storeu_ps(&y.data[8 + 32 * ioo], var6[1]);
    _mm256_storeu_ps(&y.data[16 + 32 * ioo], var6[2]);
    _mm256_storeu_ps(&y.data[24 + 32 * ioo], var6[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  var0 = _mm256_broadcast_ss(&H[0]);
  __m256 var1;
  __m256 var2;
  __m256 var3;
  __m256 var4;
  var4 = _mm256_broadcast_ss(&H[2 + 1]);
  __m256 var5;
  __m256 var6;
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    var1 = _mm256_mul_ps(var0, xReg);
    var2 = _mm256_add_ps(var1, yReg);
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var2);
    var3 = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));
    var5 = _mm256_mul_ps(var4, yReg);
    var6 = _mm256_add_ps(var3, var5);
    _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var6);
  }
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 var0;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[0]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var0, xReg);
var1 = _mm256_blendv_ps (var1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var1, yReg);
var2 = _mm256_blendv_ps (var2, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var2);
    }
    
  __m256 var3;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));;
var3 = _mm256_blendv_ps (var3, src_sign, _mm256_castsi256_ps(cmp));
}

  __m256 var4;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var4 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2 + 1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var5;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var4, yReg);
var5 = _mm256_blendv_ps (var5, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var6;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var3, var5);
var6 = _mm256_blendv_ps (var6, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var6);
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
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 var0[4];
  __m256 var1[4];
  __m256 var2[4];
  __m256 var3[4];
  __m256 var4[4];
  __m256 var5[4];
  var0[0] = _mm256_broadcast_ss(&H[1]);
  var0[1] = _mm256_broadcast_ss(&H[1]);
  var0[2] = _mm256_broadcast_ss(&H[1]);
  var0[3] = _mm256_broadcast_ss(&H[1]);
  var3[0] = _mm256_broadcast_ss(&H[2]);
  var3[1] = _mm256_broadcast_ss(&H[2]);
  var3[2] = _mm256_broadcast_ss(&H[2]);
  var3[3] = _mm256_broadcast_ss(&H[2]);
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
    xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
    xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
    yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
    yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
    var1[0] = _mm256_mul_ps(var0[0], yReg[0]);
    var1[1] = _mm256_mul_ps(var0[1], yReg[1]);
    var1[2] = _mm256_mul_ps(var0[2], yReg[2]);
    var1[3] = _mm256_mul_ps(var0[3], yReg[3]);
    var2[0] = _mm256_add_ps(xReg[0], var1[0]);
    var2[1] = _mm256_add_ps(xReg[1], var1[1]);
    var2[2] = _mm256_add_ps(xReg[2], var1[2]);
    var2[3] = _mm256_add_ps(xReg[3], var1[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], var2[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], var2[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], var2[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], var2[3]);
    var4[0] = _mm256_mul_ps(var3[0], xReg[0]);
    var4[1] = _mm256_mul_ps(var3[1], xReg[1]);
    var4[2] = _mm256_mul_ps(var3[2], xReg[2]);
    var4[3] = _mm256_mul_ps(var3[3], xReg[3]);
    var5[0] = _mm256_add_ps(var4[0], yReg[0]);
    var5[1] = _mm256_add_ps(var4[1], yReg[1]);
    var5[2] = _mm256_add_ps(var4[2], yReg[2]);
    var5[3] = _mm256_add_ps(var4[3], yReg[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], var5[0]);
    _mm256_storeu_ps(&y.data[8 + 32 * ioo], var5[1]);
    _mm256_storeu_ps(&y.data[16 + 32 * ioo], var5[2]);
    _mm256_storeu_ps(&y.data[24 + 32 * ioo], var5[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 var0;
  var0 = _mm256_broadcast_ss(&H[1]);
  __m256 var1;
  __m256 var2;
  __m256 var3;
  var3 = _mm256_broadcast_ss(&H[2]);
  __m256 var4;
  __m256 var5;
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    var1 = _mm256_mul_ps(var0, yReg);
    var2 = _mm256_add_ps(xReg, var1);
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var2);
    var4 = _mm256_mul_ps(var3, xReg);
    var5 = _mm256_add_ps(var4, yReg);
    _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], var5);
  }
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 var0;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var0, yReg);
var1 = _mm256_blendv_ps (var1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(xReg, var1);
var2 = _mm256_blendv_ps (var2, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, var2);
    }
    
  __m256 var3;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var3 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 var4;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var3, xReg);
var4 = _mm256_blendv_ps (var4, mul, _mm256_castsi256_ps(cmp));
}

  __m256 var5;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var4, yReg);
var5 = _mm256_blendv_ps (var5, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, var5);
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
mm256_add_pd(out,x,y)
{out_data} = _mm256_add_pd({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_add_ps(out,x,y)
{out_data} = _mm256_add_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_broadcast_sd(out,val)
{out_data} = _mm256_broadcast_sd(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
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
mm256_prefix_add_pd(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd({x_data}, {y_data});
{out_data} = _mm256_blendv_pd ({out_data}, add, _mm256_castsi256_pd(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_add_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, add, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_broadcast_sd(out,val,bound)

    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    {out_data} = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&{val_data}), _mm256_castsi256_pd(cmp));
    }}
    
*/

/* relying on the following instruction..."
mm256_prefix_broadcast_ss(out,val,bound)

    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {out_data} = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&{val_data}), _mm256_castsi256_ps(cmp));
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
