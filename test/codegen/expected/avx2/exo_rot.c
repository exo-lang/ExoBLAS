#include "exo_rot.h"



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
// exo_drot_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     c : f64 @DRAM,
//     s : f64 @DRAM
// )
void exo_drot_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* c, const double* s ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double s_;
s_ = *s;
double c_;
c_ = *c;
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d xReg[4];
  __m256d yReg[4];
  __m256d sReg[4];
  __m256d cReg[4];
  __m256d reg[4];
  __m256d reg_1[4];
  __m256d reg_2[4];
  __m256d reg_3[4];
  __m256d reg_4[4];
  __m256d reg_5[4];
  __m256d reg_6[4];
  sReg[0] = _mm256_broadcast_sd((&s_));
  sReg[1] = _mm256_broadcast_sd((&s_));
  sReg[2] = _mm256_broadcast_sd((&s_));
  sReg[3] = _mm256_broadcast_sd((&s_));
  cReg[0] = _mm256_broadcast_sd((&c_));
  cReg[1] = _mm256_broadcast_sd((&c_));
  cReg[2] = _mm256_broadcast_sd((&c_));
  cReg[3] = _mm256_broadcast_sd((&c_));
  reg_3[0] = _mm256_mul_pd(sReg[0], _mm256_set1_pd(-1.0f));
  reg_3[1] = _mm256_mul_pd(sReg[1], _mm256_set1_pd(-1.0f));
  reg_3[2] = _mm256_mul_pd(sReg[2], _mm256_set1_pd(-1.0f));
  reg_3[3] = _mm256_mul_pd(sReg[3], _mm256_set1_pd(-1.0f));
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
    xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
    xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
    yReg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
    yReg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
    yReg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
    yReg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
    reg[0] = _mm256_mul_pd(cReg[0], xReg[0]);
    reg[1] = _mm256_mul_pd(cReg[1], xReg[1]);
    reg[2] = _mm256_mul_pd(cReg[2], xReg[2]);
    reg[3] = _mm256_mul_pd(cReg[3], xReg[3]);
    reg_1[0] = _mm256_mul_pd(sReg[0], yReg[0]);
    reg_1[1] = _mm256_mul_pd(sReg[1], yReg[1]);
    reg_1[2] = _mm256_mul_pd(sReg[2], yReg[2]);
    reg_1[3] = _mm256_mul_pd(sReg[3], yReg[3]);
    reg_2[0] = _mm256_add_pd(reg[0], reg_1[0]);
    reg_2[1] = _mm256_add_pd(reg[1], reg_1[1]);
    reg_2[2] = _mm256_add_pd(reg[2], reg_1[2]);
    reg_2[3] = _mm256_add_pd(reg[3], reg_1[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], reg_2[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], reg_2[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], reg_2[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], reg_2[3]);
    reg_4[0] = _mm256_mul_pd(reg_3[0], xReg[0]);
    reg_4[1] = _mm256_mul_pd(reg_3[1], xReg[1]);
    reg_4[2] = _mm256_mul_pd(reg_3[2], xReg[2]);
    reg_4[3] = _mm256_mul_pd(reg_3[3], xReg[3]);
    reg_5[0] = _mm256_mul_pd(cReg[0], yReg[0]);
    reg_5[1] = _mm256_mul_pd(cReg[1], yReg[1]);
    reg_5[2] = _mm256_mul_pd(cReg[2], yReg[2]);
    reg_5[3] = _mm256_mul_pd(cReg[3], yReg[3]);
    reg_6[0] = _mm256_add_pd(reg_4[0], reg_5[0]);
    reg_6[1] = _mm256_add_pd(reg_4[1], reg_5[1]);
    reg_6[2] = _mm256_add_pd(reg_4[2], reg_5[2]);
    reg_6[3] = _mm256_add_pd(reg_4[3], reg_5[3]);
    _mm256_storeu_pd(&y.data[16 * ioo], reg_6[0]);
    _mm256_storeu_pd(&y.data[4 + 16 * ioo], reg_6[1]);
    _mm256_storeu_pd(&y.data[8 + 16 * ioo], reg_6[2]);
    _mm256_storeu_pd(&y.data[12 + 16 * ioo], reg_6[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d xReg;
  __m256d yReg;
  __m256d sReg;
  sReg = _mm256_broadcast_sd((&s_));
  __m256d cReg;
  cReg = _mm256_broadcast_sd((&c_));
  __m256d reg;
  __m256d reg_1;
  __m256d reg_2;
  __m256d reg_3;
  reg_3 = _mm256_mul_pd(sReg, _mm256_set1_pd(-1.0f));
  __m256d reg_4;
  __m256d reg_5;
  __m256d reg_6;
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    yReg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    reg = _mm256_mul_pd(cReg, xReg);
    reg_1 = _mm256_mul_pd(sReg, yReg);
    reg_2 = _mm256_add_pd(reg, reg_1);
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg_2);
    reg_4 = _mm256_mul_pd(reg_3, xReg);
    reg_5 = _mm256_mul_pd(cReg, yReg);
    reg_6 = _mm256_add_pd(reg_4, reg_5);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg_6);
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
       
  __m256d sReg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    sReg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd((&s_)), _mm256_castsi256_pd(cmp));
    }
    
  __m256d cReg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    cReg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd((&c_)), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(cReg, xReg);
reg = _mm256_blendv_pd (reg, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_1;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(sReg, yReg);
reg_1 = _mm256_blendv_pd (reg_1, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg, reg_1);
reg_2 = _mm256_blendv_pd (reg_2, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg_2);
    }
    
  __m256d reg_3;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd(sReg, _mm256_set1_pd(-1.0f));
reg_3 = _mm256_blendv_pd (reg_3, src_sign, _mm256_castsi256_pd(cmp));
}

  __m256d reg_4;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg_3, xReg);
reg_4 = _mm256_blendv_pd (reg_4, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_5;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(cReg, yReg);
reg_5 = _mm256_blendv_pd (reg_5, mul, _mm256_castsi256_pd(cmp));
}

  __m256d reg_6;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd(reg_4, reg_5);
reg_6 = _mm256_blendv_pd (reg_6, add, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_6);
    }
    
}
}

// exo_drot_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     c : f64 @DRAM,
//     s : f64 @DRAM
// )
void exo_drot_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y, const double* c, const double* s ) {
double s_;
s_ = *s;
double c_;
c_ = *c;
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = c_ * xReg + s_ * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = -s_ * xReg + c_ * y.data[i * y.strides[0]];
}
}

// exo_srot_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     c : f32 @DRAM,
//     s : f32 @DRAM
// )
void exo_srot_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* c, const float* s ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float s_;
s_ = *s;
float c_;
c_ = *c;
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 xReg[4];
  __m256 yReg[4];
  __m256 sReg[4];
  __m256 cReg[4];
  __m256 reg[4];
  __m256 reg_1[4];
  __m256 reg_2[4];
  __m256 reg_3[4];
  __m256 reg_4[4];
  __m256 reg_5[4];
  __m256 reg_6[4];
  sReg[0] = _mm256_broadcast_ss((&s_));
  sReg[1] = _mm256_broadcast_ss((&s_));
  sReg[2] = _mm256_broadcast_ss((&s_));
  sReg[3] = _mm256_broadcast_ss((&s_));
  cReg[0] = _mm256_broadcast_ss((&c_));
  cReg[1] = _mm256_broadcast_ss((&c_));
  cReg[2] = _mm256_broadcast_ss((&c_));
  cReg[3] = _mm256_broadcast_ss((&c_));
  reg_3[0] = _mm256_mul_ps(sReg[0], _mm256_set1_ps(-1.0f));
  reg_3[1] = _mm256_mul_ps(sReg[1], _mm256_set1_ps(-1.0f));
  reg_3[2] = _mm256_mul_ps(sReg[2], _mm256_set1_ps(-1.0f));
  reg_3[3] = _mm256_mul_ps(sReg[3], _mm256_set1_ps(-1.0f));
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    xReg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
    xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
    xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
    yReg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
    yReg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
    yReg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
    yReg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
    reg[0] = _mm256_mul_ps(cReg[0], xReg[0]);
    reg[1] = _mm256_mul_ps(cReg[1], xReg[1]);
    reg[2] = _mm256_mul_ps(cReg[2], xReg[2]);
    reg[3] = _mm256_mul_ps(cReg[3], xReg[3]);
    reg_1[0] = _mm256_mul_ps(sReg[0], yReg[0]);
    reg_1[1] = _mm256_mul_ps(sReg[1], yReg[1]);
    reg_1[2] = _mm256_mul_ps(sReg[2], yReg[2]);
    reg_1[3] = _mm256_mul_ps(sReg[3], yReg[3]);
    reg_2[0] = _mm256_add_ps(reg[0], reg_1[0]);
    reg_2[1] = _mm256_add_ps(reg[1], reg_1[1]);
    reg_2[2] = _mm256_add_ps(reg[2], reg_1[2]);
    reg_2[3] = _mm256_add_ps(reg[3], reg_1[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], reg_2[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], reg_2[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], reg_2[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], reg_2[3]);
    reg_4[0] = _mm256_mul_ps(reg_3[0], xReg[0]);
    reg_4[1] = _mm256_mul_ps(reg_3[1], xReg[1]);
    reg_4[2] = _mm256_mul_ps(reg_3[2], xReg[2]);
    reg_4[3] = _mm256_mul_ps(reg_3[3], xReg[3]);
    reg_5[0] = _mm256_mul_ps(cReg[0], yReg[0]);
    reg_5[1] = _mm256_mul_ps(cReg[1], yReg[1]);
    reg_5[2] = _mm256_mul_ps(cReg[2], yReg[2]);
    reg_5[3] = _mm256_mul_ps(cReg[3], yReg[3]);
    reg_6[0] = _mm256_add_ps(reg_4[0], reg_5[0]);
    reg_6[1] = _mm256_add_ps(reg_4[1], reg_5[1]);
    reg_6[2] = _mm256_add_ps(reg_4[2], reg_5[2]);
    reg_6[3] = _mm256_add_ps(reg_4[3], reg_5[3]);
    _mm256_storeu_ps(&y.data[32 * ioo], reg_6[0]);
    _mm256_storeu_ps(&y.data[8 + 32 * ioo], reg_6[1]);
    _mm256_storeu_ps(&y.data[16 + 32 * ioo], reg_6[2]);
    _mm256_storeu_ps(&y.data[24 + 32 * ioo], reg_6[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 xReg;
  __m256 yReg;
  __m256 sReg;
  sReg = _mm256_broadcast_ss((&s_));
  __m256 cReg;
  cReg = _mm256_broadcast_ss((&c_));
  __m256 reg;
  __m256 reg_1;
  __m256 reg_2;
  __m256 reg_3;
  reg_3 = _mm256_mul_ps(sReg, _mm256_set1_ps(-1.0f));
  __m256 reg_4;
  __m256 reg_5;
  __m256 reg_6;
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    xReg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    yReg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    reg = _mm256_mul_ps(cReg, xReg);
    reg_1 = _mm256_mul_ps(sReg, yReg);
    reg_2 = _mm256_add_ps(reg, reg_1);
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg_2);
    reg_4 = _mm256_mul_ps(reg_3, xReg);
    reg_5 = _mm256_mul_ps(cReg, yReg);
    reg_6 = _mm256_add_ps(reg_4, reg_5);
    _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg_6);
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

  __m256 sReg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    sReg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss((&s_)), _mm256_castsi256_ps(cmp));
    }
    
  __m256 cReg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    cReg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss((&c_)), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(cReg, xReg);
reg = _mm256_blendv_ps (reg, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_1;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(sReg, yReg);
reg_1 = _mm256_blendv_ps (reg_1, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg, reg_1);
reg_2 = _mm256_blendv_ps (reg_2, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg_2);
    }
    
  __m256 reg_3;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps(sReg, _mm256_set1_ps(-1.0f));;
reg_3 = _mm256_blendv_ps (reg_3, src_sign, _mm256_castsi256_ps(cmp));
}

  __m256 reg_4;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg_3, xReg);
reg_4 = _mm256_blendv_ps (reg_4, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_5;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(cReg, yReg);
reg_5 = _mm256_blendv_ps (reg_5, mul, _mm256_castsi256_ps(cmp));
}

  __m256 reg_6;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(reg_4, reg_5);
reg_6 = _mm256_blendv_ps (reg_6, add, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_6);
    }
    
}
}

// exo_srot_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     c : f32 @DRAM,
//     s : f32 @DRAM
// )
void exo_srot_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y, const float* c, const float* s ) {
float s_;
s_ = *s;
float c_;
c_ = *c;
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = c_ * xReg + s_ * y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = -s_ * xReg + c_ * y.data[i * y.strides[0]];
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
mm256_broadcast_sd_scalar(out,val)
{out_data} = _mm256_broadcast_sd({val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss_scalar(out,val)
{out_data} = _mm256_broadcast_ss({val_data});
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
mm256_prefix_broadcast_sd_scalar(out,val,bound)

    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    {out_data} = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd({val_data}), _mm256_castsi256_pd(cmp));
    }}
    
*/

/* relying on the following instruction..."
mm256_prefix_broadcast_ss_scalar(out,val,bound)

    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {out_data} = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss({val_data}), _mm256_castsi256_ps(cmp));
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