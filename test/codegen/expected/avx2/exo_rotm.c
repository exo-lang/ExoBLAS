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
if (0 < ((((n + 3) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d reg[4];
  __m256d reg_1[4];
  __m256d reg_2[4];
  __m256d reg_3[4];
  __m256d reg_4[4];
  __m256d reg_5[4];
  __m256d reg_6[4];
  __m256d reg_7[4];
  __m256d reg_8[4];
  __m256d reg_9[4];
  reg[0] = _mm256_broadcast_sd(&H[0]);
  reg[1] = _mm256_broadcast_sd(&H[0]);
  reg[2] = _mm256_broadcast_sd(&H[0]);
  reg[3] = _mm256_broadcast_sd(&H[0]);
  reg_2[0] = _mm256_broadcast_sd(&H[1]);
  reg_2[1] = _mm256_broadcast_sd(&H[1]);
  reg_2[2] = _mm256_broadcast_sd(&H[1]);
  reg_2[3] = _mm256_broadcast_sd(&H[1]);
  reg_5[0] = _mm256_broadcast_sd(&H[2]);
  reg_5[1] = _mm256_broadcast_sd(&H[2]);
  reg_5[2] = _mm256_broadcast_sd(&H[2]);
  reg_5[3] = _mm256_broadcast_sd(&H[2]);
  reg_7[0] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_7[1] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_7[2] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_7[3] = _mm256_broadcast_sd(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 3) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[16 * ioo + 4]);
    xReg[2] = _mm256_loadu_pd(&x.data[16 * ioo + 8]);
    xReg[3] = _mm256_loadu_pd(&x.data[16 * ioo + 12]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[16 * ioo + 4]);
    yReg[2] = _mm256_loadu_pd(&y.data[16 * ioo + 8]);
    yReg[3] = _mm256_loadu_pd(&y.data[16 * ioo + 12]);
    reg_1[0] = _mm256_mul_pd(reg[0], xReg[0]);
    reg_1[1] = _mm256_mul_pd(reg[1], xReg[1]);
    reg_1[2] = _mm256_mul_pd(reg[2], xReg[2]);
    reg_1[3] = _mm256_mul_pd(reg[3], xReg[3]);
    reg_3[0] = _mm256_mul_pd(reg_2[0], yReg[0]);
    reg_3[1] = _mm256_mul_pd(reg_2[1], yReg[1]);
    reg_3[2] = _mm256_mul_pd(reg_2[2], yReg[2]);
    reg_3[3] = _mm256_mul_pd(reg_2[3], yReg[3]);
    reg_4[0] = _mm256_add_pd(reg_1[0], reg_3[0]);
    reg_4[1] = _mm256_add_pd(reg_1[1], reg_3[1]);
    reg_4[2] = _mm256_add_pd(reg_1[2], reg_3[2]);
    reg_4[3] = _mm256_add_pd(reg_1[3], reg_3[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], reg_4[0]);
    _mm256_storeu_pd(&x.data[16 * ioo + 4], reg_4[1]);
    _mm256_storeu_pd(&x.data[16 * ioo + 8], reg_4[2]);
    _mm256_storeu_pd(&x.data[16 * ioo + 12], reg_4[3]);
    reg_6[0] = _mm256_mul_pd(reg_5[0], xReg[0]);
    reg_6[1] = _mm256_mul_pd(reg_5[1], xReg[1]);
    reg_6[2] = _mm256_mul_pd(reg_5[2], xReg[2]);
    reg_6[3] = _mm256_mul_pd(reg_5[3], xReg[3]);
    reg_8[0] = _mm256_mul_pd(reg_7[0], yReg[0]);
    reg_8[1] = _mm256_mul_pd(reg_7[1], yReg[1]);
    reg_8[2] = _mm256_mul_pd(reg_7[2], yReg[2]);
    reg_8[3] = _mm256_mul_pd(reg_7[3], yReg[3]);
    reg_9[0] = _mm256_add_pd(reg_6[0], reg_8[0]);
    reg_9[1] = _mm256_add_pd(reg_6[1], reg_8[1]);
    reg_9[2] = _mm256_add_pd(reg_6[2], reg_8[2]);
    reg_9[3] = _mm256_add_pd(reg_6[3], reg_8[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], reg_9[0]);
    _mm256_storeu_pd(&y.data[16 * ioo + 4], reg_9[1]);
    _mm256_storeu_pd(&y.data[16 * ioo + 8], reg_9[2]);
    _mm256_storeu_pd(&y.data[16 * ioo + 12], reg_9[3]);
  }
}
if (0 < (((n + 3) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d reg;
  reg = _mm256_broadcast_sd(&H[0]);
  __m256d reg_1;
  __m256d reg_2;
  reg_2 = _mm256_broadcast_sd(&H[1]);
  __m256d reg_3;
  __m256d reg_4;
  __m256d reg_5;
  reg_5 = _mm256_broadcast_sd(&H[2]);
  __m256d reg_6;
  __m256d reg_7;
  reg_7 = _mm256_broadcast_sd(&H[2 + 1]);
  __m256d reg_8;
  __m256d reg_9;
  for (int_fast32_t ioi = 0; ioi < (((n + 3) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    reg_1 = _mm256_mul_pd(reg, xReg);
    reg_3 = _mm256_mul_pd(reg_2, yReg);
    reg_4 = _mm256_add_pd(reg_1, reg_3);
    _mm256_storeu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_4);
    reg_6 = _mm256_mul_pd(reg_5, xReg);
    reg_8 = _mm256_mul_pd(reg_7, yReg);
    reg_9 = _mm256_add_pd(reg_6, reg_8);
    _mm256_storeu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_9);
  }
}
for (int_fast32_t io = ((n + 3) / (4)) - 1; io < ((n + 3) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d reg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[0]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg, xReg);
reg_1 = _mm256_blendv_pd (reg_1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_2;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg_2 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_3;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_2, yReg);
reg_3 = _mm256_blendv_pd (reg_3, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_4;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_1, reg_3);
reg_4 = _mm256_blendv_pd (reg_4, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg_4);
    }
    
  __m256d reg_5;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg_5 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_6;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_5, xReg);
reg_6 = _mm256_blendv_pd (reg_6, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_7;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg_7 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2 + 1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_8;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_7, yReg);
reg_8 = _mm256_blendv_pd (reg_8, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_9;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_6, reg_8);
reg_9 = _mm256_blendv_pd (reg_9, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_9);
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
if (0 < ((((n + 3) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d reg[4];
  __m256d reg_1[4];
  __m256d reg_2[4];
  __m256d reg_3[4];
  __m256d reg_4[4];
  __m256d reg_5[4];
  __m256d reg_6[4];
  reg[0] = _mm256_broadcast_sd(&H[0]);
  reg[1] = _mm256_broadcast_sd(&H[0]);
  reg[2] = _mm256_broadcast_sd(&H[0]);
  reg[3] = _mm256_broadcast_sd(&H[0]);
  reg_4[0] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_4[1] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_4[2] = _mm256_broadcast_sd(&H[2 + 1]);
  reg_4[3] = _mm256_broadcast_sd(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 3) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[16 * ioo + 4]);
    xReg[2] = _mm256_loadu_pd(&x.data[16 * ioo + 8]);
    xReg[3] = _mm256_loadu_pd(&x.data[16 * ioo + 12]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[16 * ioo + 4]);
    yReg[2] = _mm256_loadu_pd(&y.data[16 * ioo + 8]);
    yReg[3] = _mm256_loadu_pd(&y.data[16 * ioo + 12]);
    reg_1[0] = _mm256_mul_pd(reg[0], xReg[0]);
    reg_1[1] = _mm256_mul_pd(reg[1], xReg[1]);
    reg_1[2] = _mm256_mul_pd(reg[2], xReg[2]);
    reg_1[3] = _mm256_mul_pd(reg[3], xReg[3]);
    reg_2[0] = _mm256_add_pd(reg_1[0], yReg[0]);
    reg_2[1] = _mm256_add_pd(reg_1[1], yReg[1]);
    reg_2[2] = _mm256_add_pd(reg_1[2], yReg[2]);
    reg_2[3] = _mm256_add_pd(reg_1[3], yReg[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], reg_2[0]);
    _mm256_storeu_pd(&x.data[16 * ioo + 4], reg_2[1]);
    _mm256_storeu_pd(&x.data[16 * ioo + 8], reg_2[2]);
    _mm256_storeu_pd(&x.data[16 * ioo + 12], reg_2[3]);
    reg_3[0] = _mm256_mul_pd(xReg[0], _mm256_set1_pd(-1.0f));
    reg_3[1] = _mm256_mul_pd(xReg[1], _mm256_set1_pd(-1.0f));
    reg_3[2] = _mm256_mul_pd(xReg[2], _mm256_set1_pd(-1.0f));
    reg_3[3] = _mm256_mul_pd(xReg[3], _mm256_set1_pd(-1.0f));
    reg_5[0] = _mm256_mul_pd(reg_4[0], yReg[0]);
    reg_5[1] = _mm256_mul_pd(reg_4[1], yReg[1]);
    reg_5[2] = _mm256_mul_pd(reg_4[2], yReg[2]);
    reg_5[3] = _mm256_mul_pd(reg_4[3], yReg[3]);
    reg_6[0] = _mm256_add_pd(reg_3[0], reg_5[0]);
    reg_6[1] = _mm256_add_pd(reg_3[1], reg_5[1]);
    reg_6[2] = _mm256_add_pd(reg_3[2], reg_5[2]);
    reg_6[3] = _mm256_add_pd(reg_3[3], reg_5[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], reg_6[0]);
    _mm256_storeu_pd(&y.data[16 * ioo + 4], reg_6[1]);
    _mm256_storeu_pd(&y.data[16 * ioo + 8], reg_6[2]);
    _mm256_storeu_pd(&y.data[16 * ioo + 12], reg_6[3]);
  }
}
if (0 < (((n + 3) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d reg;
  reg = _mm256_broadcast_sd(&H[0]);
  __m256d reg_1;
  __m256d reg_2;
  __m256d reg_3;
  __m256d reg_4;
  reg_4 = _mm256_broadcast_sd(&H[2 + 1]);
  __m256d reg_5;
  __m256d reg_6;
  for (int_fast32_t ioi = 0; ioi < (((n + 3) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    reg_1 = _mm256_mul_pd(reg, xReg);
    reg_2 = _mm256_add_pd(reg_1, yReg);
    _mm256_storeu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_2);
    reg_3 = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
    reg_5 = _mm256_mul_pd(reg_4, yReg);
    reg_6 = _mm256_add_pd(reg_3, reg_5);
    _mm256_storeu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_6);
  }
}
for (int_fast32_t io = ((n + 3) / (4)) - 1; io < ((n + 3) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d reg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[0]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg, xReg);
reg_1 = _mm256_blendv_pd (reg_1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_1, yReg);
reg_2 = _mm256_blendv_pd (reg_2, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg_2);
    }
    
  __m256d reg_3;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd(xReg, _mm256_set1_pd(-1.0f));
reg_3 = _mm256_blendv_pd (reg_3, src_sign, _mm256_castsi256_pd(cmp));
}

  __m256d reg_4;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg_4 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2 + 1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_5;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_4, yReg);
reg_5 = _mm256_blendv_pd (reg_5, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_6;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_3, reg_5);
reg_6 = _mm256_blendv_pd (reg_6, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_6);
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
if (0 < ((((n + 3) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d reg[4];
  __m256d reg_1[4];
  __m256d reg_2[4];
  __m256d reg_3[4];
  __m256d reg_4[4];
  __m256d reg_5[4];
  reg[0] = _mm256_broadcast_sd(&H[1]);
  reg[1] = _mm256_broadcast_sd(&H[1]);
  reg[2] = _mm256_broadcast_sd(&H[1]);
  reg[3] = _mm256_broadcast_sd(&H[1]);
  reg_3[0] = _mm256_broadcast_sd(&H[2]);
  reg_3[1] = _mm256_broadcast_sd(&H[2]);
  reg_3[2] = _mm256_broadcast_sd(&H[2]);
  reg_3[3] = _mm256_broadcast_sd(&H[2]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 3) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[16 * ioo + 4]);
    xReg[2] = _mm256_loadu_pd(&x.data[16 * ioo + 8]);
    xReg[3] = _mm256_loadu_pd(&x.data[16 * ioo + 12]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[16 * ioo + 4]);
    yReg[2] = _mm256_loadu_pd(&y.data[16 * ioo + 8]);
    yReg[3] = _mm256_loadu_pd(&y.data[16 * ioo + 12]);
    reg_1[0] = _mm256_mul_pd(reg[0], yReg[0]);
    reg_1[1] = _mm256_mul_pd(reg[1], yReg[1]);
    reg_1[2] = _mm256_mul_pd(reg[2], yReg[2]);
    reg_1[3] = _mm256_mul_pd(reg[3], yReg[3]);
    reg_2[0] = _mm256_add_pd(xReg[0], reg_1[0]);
    reg_2[1] = _mm256_add_pd(xReg[1], reg_1[1]);
    reg_2[2] = _mm256_add_pd(xReg[2], reg_1[2]);
    reg_2[3] = _mm256_add_pd(xReg[3], reg_1[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], reg_2[0]);
    _mm256_storeu_pd(&x.data[16 * ioo + 4], reg_2[1]);
    _mm256_storeu_pd(&x.data[16 * ioo + 8], reg_2[2]);
    _mm256_storeu_pd(&x.data[16 * ioo + 12], reg_2[3]);
    reg_4[0] = _mm256_mul_pd(reg_3[0], xReg[0]);
    reg_4[1] = _mm256_mul_pd(reg_3[1], xReg[1]);
    reg_4[2] = _mm256_mul_pd(reg_3[2], xReg[2]);
    reg_4[3] = _mm256_mul_pd(reg_3[3], xReg[3]);
    reg_5[0] = _mm256_add_pd(reg_4[0], yReg[0]);
    reg_5[1] = _mm256_add_pd(reg_4[1], yReg[1]);
    reg_5[2] = _mm256_add_pd(reg_4[2], yReg[2]);
    reg_5[3] = _mm256_add_pd(reg_4[3], yReg[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], reg_5[0]);
    _mm256_storeu_pd(&y.data[16 * ioo + 4], reg_5[1]);
    _mm256_storeu_pd(&y.data[16 * ioo + 8], reg_5[2]);
    _mm256_storeu_pd(&y.data[16 * ioo + 12], reg_5[3]);
  }
}
if (0 < (((n + 3) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d reg;
  reg = _mm256_broadcast_sd(&H[1]);
  __m256d reg_1;
  __m256d reg_2;
  __m256d reg_3;
  reg_3 = _mm256_broadcast_sd(&H[2]);
  __m256d reg_4;
  __m256d reg_5;
  for (int_fast32_t ioi = 0; ioi < (((n + 3) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi]);
    reg_1 = _mm256_mul_pd(reg, yReg);
    reg_2 = _mm256_add_pd(xReg, reg_1);
    _mm256_storeu_pd(&x.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_2);
    reg_4 = _mm256_mul_pd(reg_3, xReg);
    reg_5 = _mm256_add_pd(reg_4, yReg);
    _mm256_storeu_pd(&y.data[16 * ((((n + 3) / 4) - 1) / 4) + 4 * ioi], reg_5);
  }
}
for (int_fast32_t io = ((n + 3) / (4)) - 1; io < ((n + 3) / (4)); io++) {
  __m256d xReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xReg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d yReg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            yReg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  __m256d reg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[1]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg, yReg);
reg_1 = _mm256_blendv_pd (reg_1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(xReg, reg_1);
reg_2 = _mm256_blendv_pd (reg_2, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg_2);
    }
    
  __m256d reg_3;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg_3 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&H[2]), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_4;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_3, xReg);
reg_4 = _mm256_blendv_pd (reg_4, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_5;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_4, yReg);
reg_5 = _mm256_blendv_pd (reg_5, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-4 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_5);
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
if (0 < ((((n + 7) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 reg[4];
  __m256 reg_1[4];
  __m256 reg_2[4];
  __m256 reg_3[4];
  __m256 reg_4[4];
  __m256 reg_5[4];
  __m256 reg_6[4];
  __m256 reg_7[4];
  __m256 reg_8[4];
  __m256 reg_9[4];
  reg[0] = _mm256_broadcast_ss(&H[0]);
  reg[1] = _mm256_broadcast_ss(&H[0]);
  reg[2] = _mm256_broadcast_ss(&H[0]);
  reg[3] = _mm256_broadcast_ss(&H[0]);
  reg_2[0] = _mm256_broadcast_ss(&H[1]);
  reg_2[1] = _mm256_broadcast_ss(&H[1]);
  reg_2[2] = _mm256_broadcast_ss(&H[1]);
  reg_2[3] = _mm256_broadcast_ss(&H[1]);
  reg_5[0] = _mm256_broadcast_ss(&H[2]);
  reg_5[1] = _mm256_broadcast_ss(&H[2]);
  reg_5[2] = _mm256_broadcast_ss(&H[2]);
  reg_5[3] = _mm256_broadcast_ss(&H[2]);
  reg_7[0] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_7[1] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_7[2] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_7[3] = _mm256_broadcast_ss(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 7) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[32 * ioo + 8]);
    xReg[2] = _mm256_loadu_ps(&x.data[32 * ioo + 16]);
    xReg[3] = _mm256_loadu_ps(&x.data[32 * ioo + 24]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[32 * ioo + 8]);
    yReg[2] = _mm256_loadu_ps(&y.data[32 * ioo + 16]);
    yReg[3] = _mm256_loadu_ps(&y.data[32 * ioo + 24]);
    reg_1[0] = _mm256_mul_ps(reg[0], xReg[0]);
    reg_1[1] = _mm256_mul_ps(reg[1], xReg[1]);
    reg_1[2] = _mm256_mul_ps(reg[2], xReg[2]);
    reg_1[3] = _mm256_mul_ps(reg[3], xReg[3]);
    reg_3[0] = _mm256_mul_ps(reg_2[0], yReg[0]);
    reg_3[1] = _mm256_mul_ps(reg_2[1], yReg[1]);
    reg_3[2] = _mm256_mul_ps(reg_2[2], yReg[2]);
    reg_3[3] = _mm256_mul_ps(reg_2[3], yReg[3]);
    reg_4[0] = _mm256_add_ps(reg_1[0], reg_3[0]);
    reg_4[1] = _mm256_add_ps(reg_1[1], reg_3[1]);
    reg_4[2] = _mm256_add_ps(reg_1[2], reg_3[2]);
    reg_4[3] = _mm256_add_ps(reg_1[3], reg_3[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], reg_4[0]);
    _mm256_storeu_ps(&x.data[32 * ioo + 8], reg_4[1]);
    _mm256_storeu_ps(&x.data[32 * ioo + 16], reg_4[2]);
    _mm256_storeu_ps(&x.data[32 * ioo + 24], reg_4[3]);
    reg_6[0] = _mm256_mul_ps(reg_5[0], xReg[0]);
    reg_6[1] = _mm256_mul_ps(reg_5[1], xReg[1]);
    reg_6[2] = _mm256_mul_ps(reg_5[2], xReg[2]);
    reg_6[3] = _mm256_mul_ps(reg_5[3], xReg[3]);
    reg_8[0] = _mm256_mul_ps(reg_7[0], yReg[0]);
    reg_8[1] = _mm256_mul_ps(reg_7[1], yReg[1]);
    reg_8[2] = _mm256_mul_ps(reg_7[2], yReg[2]);
    reg_8[3] = _mm256_mul_ps(reg_7[3], yReg[3]);
    reg_9[0] = _mm256_add_ps(reg_6[0], reg_8[0]);
    reg_9[1] = _mm256_add_ps(reg_6[1], reg_8[1]);
    reg_9[2] = _mm256_add_ps(reg_6[2], reg_8[2]);
    reg_9[3] = _mm256_add_ps(reg_6[3], reg_8[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], reg_9[0]);
    _mm256_storeu_ps(&y.data[32 * ioo + 8], reg_9[1]);
    _mm256_storeu_ps(&y.data[32 * ioo + 16], reg_9[2]);
    _mm256_storeu_ps(&y.data[32 * ioo + 24], reg_9[3]);
  }
}
if (0 < (((n + 7) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 reg;
  reg = _mm256_broadcast_ss(&H[0]);
  __m256 reg_1;
  __m256 reg_2;
  reg_2 = _mm256_broadcast_ss(&H[1]);
  __m256 reg_3;
  __m256 reg_4;
  __m256 reg_5;
  reg_5 = _mm256_broadcast_ss(&H[2]);
  __m256 reg_6;
  __m256 reg_7;
  reg_7 = _mm256_broadcast_ss(&H[2 + 1]);
  __m256 reg_8;
  __m256 reg_9;
  for (int_fast32_t ioi = 0; ioi < (((n + 7) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    reg_1 = _mm256_mul_ps(reg, xReg);
    reg_3 = _mm256_mul_ps(reg_2, yReg);
    reg_4 = _mm256_add_ps(reg_1, reg_3);
    _mm256_storeu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_4);
    reg_6 = _mm256_mul_ps(reg_5, xReg);
    reg_8 = _mm256_mul_ps(reg_7, yReg);
    reg_9 = _mm256_add_ps(reg_6, reg_8);
    _mm256_storeu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_9);
  }
}
for (int_fast32_t io = ((n + 7) / (8)) - 1; io < ((n + 7) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 reg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[0]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg, xReg);
reg_1 = _mm256_blendv_ps (reg_1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_2;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_2 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_3;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_2, yReg);
reg_3 = _mm256_blendv_ps (reg_3, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_4;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_1, reg_3);
reg_4 = _mm256_blendv_ps (reg_4, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg_4);
    }
    
  __m256 reg_5;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_5 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_6;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_5, xReg);
reg_6 = _mm256_blendv_ps (reg_6, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_7;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_7 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2 + 1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_8;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_7, yReg);
reg_8 = _mm256_blendv_ps (reg_8, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_9;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_6, reg_8);
reg_9 = _mm256_blendv_ps (reg_9, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_9);
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
if (0 < ((((n + 7) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 reg[4];
  __m256 reg_1[4];
  __m256 reg_2[4];
  __m256 reg_3[4];
  __m256 reg_4[4];
  __m256 reg_5[4];
  __m256 reg_6[4];
  reg[0] = _mm256_broadcast_ss(&H[0]);
  reg[1] = _mm256_broadcast_ss(&H[0]);
  reg[2] = _mm256_broadcast_ss(&H[0]);
  reg[3] = _mm256_broadcast_ss(&H[0]);
  reg_4[0] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_4[1] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_4[2] = _mm256_broadcast_ss(&H[2 + 1]);
  reg_4[3] = _mm256_broadcast_ss(&H[2 + 1]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 7) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[32 * ioo + 8]);
    xReg[2] = _mm256_loadu_ps(&x.data[32 * ioo + 16]);
    xReg[3] = _mm256_loadu_ps(&x.data[32 * ioo + 24]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[32 * ioo + 8]);
    yReg[2] = _mm256_loadu_ps(&y.data[32 * ioo + 16]);
    yReg[3] = _mm256_loadu_ps(&y.data[32 * ioo + 24]);
    reg_1[0] = _mm256_mul_ps(reg[0], xReg[0]);
    reg_1[1] = _mm256_mul_ps(reg[1], xReg[1]);
    reg_1[2] = _mm256_mul_ps(reg[2], xReg[2]);
    reg_1[3] = _mm256_mul_ps(reg[3], xReg[3]);
    reg_2[0] = _mm256_add_ps(reg_1[0], yReg[0]);
    reg_2[1] = _mm256_add_ps(reg_1[1], yReg[1]);
    reg_2[2] = _mm256_add_ps(reg_1[2], yReg[2]);
    reg_2[3] = _mm256_add_ps(reg_1[3], yReg[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], reg_2[0]);
    _mm256_storeu_ps(&x.data[32 * ioo + 8], reg_2[1]);
    _mm256_storeu_ps(&x.data[32 * ioo + 16], reg_2[2]);
    _mm256_storeu_ps(&x.data[32 * ioo + 24], reg_2[3]);
    reg_3[0] = _mm256_mul_ps(xReg[0], _mm256_set1_ps(-1.0f));
    reg_3[1] = _mm256_mul_ps(xReg[1], _mm256_set1_ps(-1.0f));
    reg_3[2] = _mm256_mul_ps(xReg[2], _mm256_set1_ps(-1.0f));
    reg_3[3] = _mm256_mul_ps(xReg[3], _mm256_set1_ps(-1.0f));
    reg_5[0] = _mm256_mul_ps(reg_4[0], yReg[0]);
    reg_5[1] = _mm256_mul_ps(reg_4[1], yReg[1]);
    reg_5[2] = _mm256_mul_ps(reg_4[2], yReg[2]);
    reg_5[3] = _mm256_mul_ps(reg_4[3], yReg[3]);
    reg_6[0] = _mm256_add_ps(reg_3[0], reg_5[0]);
    reg_6[1] = _mm256_add_ps(reg_3[1], reg_5[1]);
    reg_6[2] = _mm256_add_ps(reg_3[2], reg_5[2]);
    reg_6[3] = _mm256_add_ps(reg_3[3], reg_5[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], reg_6[0]);
    _mm256_storeu_ps(&y.data[32 * ioo + 8], reg_6[1]);
    _mm256_storeu_ps(&y.data[32 * ioo + 16], reg_6[2]);
    _mm256_storeu_ps(&y.data[32 * ioo + 24], reg_6[3]);
  }
}
if (0 < (((n + 7) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 reg;
  reg = _mm256_broadcast_ss(&H[0]);
  __m256 reg_1;
  __m256 reg_2;
  __m256 reg_3;
  __m256 reg_4;
  reg_4 = _mm256_broadcast_ss(&H[2 + 1]);
  __m256 reg_5;
  __m256 reg_6;
  for (int_fast32_t ioi = 0; ioi < (((n + 7) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    reg_1 = _mm256_mul_ps(reg, xReg);
    reg_2 = _mm256_add_ps(reg_1, yReg);
    _mm256_storeu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_2);
    reg_3 = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));
    reg_5 = _mm256_mul_ps(reg_4, yReg);
    reg_6 = _mm256_add_ps(reg_3, reg_5);
    _mm256_storeu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_6);
  }
}
for (int_fast32_t io = ((n + 7) / (8)) - 1; io < ((n + 7) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 reg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[0]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg, xReg);
reg_1 = _mm256_blendv_ps (reg_1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_1, yReg);
reg_2 = _mm256_blendv_ps (reg_2, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg_2);
    }
    
  __m256 reg_3;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps(xReg, _mm256_set1_ps(-1.0f));;
reg_3 = _mm256_blendv_ps (reg_3, src_sign, _mm256_castsi256_ps(cmp));
}

  __m256 reg_4;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_4 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2 + 1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_5;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_4, yReg);
reg_5 = _mm256_blendv_ps (reg_5, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_6;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_3, reg_5);
reg_6 = _mm256_blendv_ps (reg_6, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_6);
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
if (0 < ((((n + 7) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 reg[4];
  __m256 reg_1[4];
  __m256 reg_2[4];
  __m256 reg_3[4];
  __m256 reg_4[4];
  __m256 reg_5[4];
  reg[0] = _mm256_broadcast_ss(&H[1]);
  reg[1] = _mm256_broadcast_ss(&H[1]);
  reg[2] = _mm256_broadcast_ss(&H[1]);
  reg[3] = _mm256_broadcast_ss(&H[1]);
  reg_3[0] = _mm256_broadcast_ss(&H[2]);
  reg_3[1] = _mm256_broadcast_ss(&H[2]);
  reg_3[2] = _mm256_broadcast_ss(&H[2]);
  reg_3[3] = _mm256_broadcast_ss(&H[2]);
  for (int_fast32_t ioo = 0; ioo < ((((n + 7) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[32 * ioo + 8]);
    xReg[2] = _mm256_loadu_ps(&x.data[32 * ioo + 16]);
    xReg[3] = _mm256_loadu_ps(&x.data[32 * ioo + 24]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[32 * ioo + 8]);
    yReg[2] = _mm256_loadu_ps(&y.data[32 * ioo + 16]);
    yReg[3] = _mm256_loadu_ps(&y.data[32 * ioo + 24]);
    reg_1[0] = _mm256_mul_ps(reg[0], yReg[0]);
    reg_1[1] = _mm256_mul_ps(reg[1], yReg[1]);
    reg_1[2] = _mm256_mul_ps(reg[2], yReg[2]);
    reg_1[3] = _mm256_mul_ps(reg[3], yReg[3]);
    reg_2[0] = _mm256_add_ps(xReg[0], reg_1[0]);
    reg_2[1] = _mm256_add_ps(xReg[1], reg_1[1]);
    reg_2[2] = _mm256_add_ps(xReg[2], reg_1[2]);
    reg_2[3] = _mm256_add_ps(xReg[3], reg_1[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], reg_2[0]);
    _mm256_storeu_ps(&x.data[32 * ioo + 8], reg_2[1]);
    _mm256_storeu_ps(&x.data[32 * ioo + 16], reg_2[2]);
    _mm256_storeu_ps(&x.data[32 * ioo + 24], reg_2[3]);
    reg_4[0] = _mm256_mul_ps(reg_3[0], xReg[0]);
    reg_4[1] = _mm256_mul_ps(reg_3[1], xReg[1]);
    reg_4[2] = _mm256_mul_ps(reg_3[2], xReg[2]);
    reg_4[3] = _mm256_mul_ps(reg_3[3], xReg[3]);
    reg_5[0] = _mm256_add_ps(reg_4[0], yReg[0]);
    reg_5[1] = _mm256_add_ps(reg_4[1], yReg[1]);
    reg_5[2] = _mm256_add_ps(reg_4[2], yReg[2]);
    reg_5[3] = _mm256_add_ps(reg_4[3], yReg[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], reg_5[0]);
    _mm256_storeu_ps(&y.data[32 * ioo + 8], reg_5[1]);
    _mm256_storeu_ps(&y.data[32 * ioo + 16], reg_5[2]);
    _mm256_storeu_ps(&y.data[32 * ioo + 24], reg_5[3]);
  }
}
if (0 < (((n + 7) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 reg;
  reg = _mm256_broadcast_ss(&H[1]);
  __m256 reg_1;
  __m256 reg_2;
  __m256 reg_3;
  reg_3 = _mm256_broadcast_ss(&H[2]);
  __m256 reg_4;
  __m256 reg_5;
  for (int_fast32_t ioi = 0; ioi < (((n + 7) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi]);
    reg_1 = _mm256_mul_ps(reg, yReg);
    reg_2 = _mm256_add_ps(xReg, reg_1);
    _mm256_storeu_ps(&x.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_2);
    reg_4 = _mm256_mul_ps(reg_3, xReg);
    reg_5 = _mm256_add_ps(reg_4, yReg);
    _mm256_storeu_ps(&y.data[32 * ((((n + 7) / 8) - 1) / 4) + 8 * ioi], reg_5);
  }
}
for (int_fast32_t io = ((n + 7) / (8)) - 1; io < ((n + 7) / (8)); io++) {
  __m256 xReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xReg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 yReg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    yReg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  __m256 reg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[1]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg, yReg);
reg_1 = _mm256_blendv_ps (reg_1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(xReg, reg_1);
reg_2 = _mm256_blendv_ps (reg_2, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg_2);
    }
    
  __m256 reg_3;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_3 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&H[2]), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_4;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_3, xReg);
reg_4 = _mm256_blendv_ps (reg_4, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_5;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_4, yReg);
reg_5 = _mm256_blendv_ps (reg_5, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * io + n + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_5);
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
