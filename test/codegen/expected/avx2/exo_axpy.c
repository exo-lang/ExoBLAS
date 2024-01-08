#include "exo_axpy.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction..."
avx2_prefix_reduce_add_wide_pd(dst,src,bound)

    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src = _mm256_blendv_pd (_mm256_setzero_pd(), {src_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_add_pd(prefixed_src, {dst_data});
    }}
    
*/

/* relying on the following instruction..."
avx2_prefix_reduce_add_wide_ps(dst,src,bound)

    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src = _mm256_blendv_ps (_mm256_setzero_ps(), {src_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_add_ps(prefixed_src, {dst_data});
    }}
    
*/

/* relying on the following instruction..."
avx2_reduce_add_wide_pd(dst,src)
{dst_data} = _mm256_add_pd({src_data}, {dst_data});
*/

/* relying on the following instruction..."
avx2_reduce_add_wide_ps(dst,src)
{dst_data} = _mm256_add_ps({src_data}, {dst_data});
*/
// exo_daxpy_alpha_1_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_alpha_1_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d reg[4];
  __m256d reg_1[4];
  reg[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  reg[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  reg[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  reg[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  reg_1[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  reg_1[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  reg_1[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  reg_1[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  reg_1[0] = _mm256_add_pd(reg[0], reg_1[0]);
  reg_1[1] = _mm256_add_pd(reg[1], reg_1[1]);
  reg_1[2] = _mm256_add_pd(reg[2], reg_1[2]);
  reg_1[3] = _mm256_add_pd(reg[3], reg_1[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], reg_1[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], reg_1[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], reg_1[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], reg_1[3]);
}
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d reg;
  reg = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  __m256d reg_1;
  reg_1 = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  reg_1 = _mm256_add_pd(reg, reg_1);
  _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg_1);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d reg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            reg = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d reg_1;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            reg_1 = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src = _mm256_blendv_pd (_mm256_setzero_pd(), reg, _mm256_castsi256_pd(cmp));
    reg_1 = _mm256_add_pd(prefixed_src, reg_1);
    }
    
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_1);
    }
    
}
}

// exo_daxpy_alpha_1_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_alpha_1_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] += x.data[i * x.strides[0]];
}
}

// exo_daxpy_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double alpha_;
alpha_ = *alpha;
__m256d reg[4];
__m256d reg_1[4];
__m256d reg_2[4];
if (((((3 + n) / (4)) - 1) / (4)) > 0) {
  reg_2[0] = _mm256_broadcast_sd((&alpha_));
}
if (((((3 + n) / (4)) - 1) / (4)) > 0) {
  reg_2[1] = _mm256_broadcast_sd((&alpha_));
}
if (((((3 + n) / (4)) - 1) / (4)) > 0) {
  reg_2[2] = _mm256_broadcast_sd((&alpha_));
}
if (((((3 + n) / (4)) - 1) / (4)) > 0) {
  reg_2[3] = _mm256_broadcast_sd((&alpha_));
}
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  reg_1[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  reg_1[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  reg_1[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  reg_1[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  reg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  reg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  reg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  reg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  reg[0] = _mm256_fmadd_pd(reg_2[0], reg_1[0], reg[0]);
  reg[1] = _mm256_fmadd_pd(reg_2[1], reg_1[1], reg[1]);
  reg[2] = _mm256_fmadd_pd(reg_2[2], reg_1[2], reg[2]);
  reg[3] = _mm256_fmadd_pd(reg_2[3], reg_1[3], reg[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], reg[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], reg[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], reg[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], reg[3]);
}
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    __m256d reg_3;
    reg_3 = _mm256_broadcast_sd((&alpha_));
    __m256d reg_4;
    reg_4 = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    __m256d reg_5;
    reg_5 = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    reg_5 = _mm256_fmadd_pd(reg_3, reg_4, reg_5);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg_5);
  }
  for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
      __m256d reg_3;
    
      {
      __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
      __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
      __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
      reg_3 = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd((&alpha_)), _mm256_castsi256_pd(cmp));
      }
      
      __m256d reg_4;
    
      {
          __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
          __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
          __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
          reg_4 = _mm256_maskload_pd(&x.data[4 * io], cmp);
      }
        
      __m256d reg_5;
    
      {
          __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
          __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
          __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
          reg_5 = _mm256_maskload_pd(&y.data[4 * io], cmp);
      }
        
    
      {
          __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
          __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
          __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
          __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), reg_3, _mm256_castsi256_pd(cmp));
          reg_5 = _mm256_fmadd_pd(prefixed_src1, reg_4, reg_5);
      }

    
      {
      __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
      __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
      __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
      _mm256_maskstore_pd(&y.data[4 * io], cmp, reg_5);
      }
      
  }
}

// exo_daxpy_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_daxpy_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] += alpha_ * x.data[i * x.strides[0]];
}
}

// exo_saxpy_alpha_1_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_alpha_1_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 reg[4];
  __m256 reg_1[4];
  reg[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  reg[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  reg[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  reg[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  reg_1[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  reg_1[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  reg_1[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  reg_1[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  reg_1[0] = _mm256_add_ps(reg[0], reg_1[0]);
  reg_1[1] = _mm256_add_ps(reg[1], reg_1[1]);
  reg_1[2] = _mm256_add_ps(reg[2], reg_1[2]);
  reg_1[3] = _mm256_add_ps(reg[3], reg_1[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], reg_1[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], reg_1[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], reg_1[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], reg_1[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 reg;
  reg = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  __m256 reg_1;
  reg_1 = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  reg_1 = _mm256_add_ps(reg, reg_1);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg_1);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 reg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 reg_1;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_1 = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src = _mm256_blendv_ps (_mm256_setzero_ps(), reg, _mm256_castsi256_ps(cmp));
    reg_1 = _mm256_add_ps(prefixed_src, reg_1);
    }
    
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_1);
    }
    
}
}

// exo_saxpy_alpha_1_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_alpha_1_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] += x.data[i * x.strides[0]];
}
}

// exo_saxpy_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float alpha_;
alpha_ = *alpha;
__m256 reg[4];
__m256 reg_1[4];
__m256 reg_2[4];
if (((((7 + n) / (8)) - 1) / (4)) > 0) {
  reg_2[0] = _mm256_broadcast_ss((&alpha_));
}
if (((((7 + n) / (8)) - 1) / (4)) > 0) {
  reg_2[1] = _mm256_broadcast_ss((&alpha_));
}
if (((((7 + n) / (8)) - 1) / (4)) > 0) {
  reg_2[2] = _mm256_broadcast_ss((&alpha_));
}
if (((((7 + n) / (8)) - 1) / (4)) > 0) {
  reg_2[3] = _mm256_broadcast_ss((&alpha_));
}
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  reg_1[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  reg_1[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  reg_1[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  reg_1[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  reg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  reg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  reg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  reg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  reg[0] = _mm256_fmadd_ps(reg_2[0], reg_1[0], reg[0]);
  reg[1] = _mm256_fmadd_ps(reg_2[1], reg_1[1], reg[1]);
  reg[2] = _mm256_fmadd_ps(reg_2[2], reg_1[2], reg[2]);
  reg[3] = _mm256_fmadd_ps(reg_2[3], reg_1[3], reg[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], reg[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], reg[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], reg[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], reg[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 reg_3;
  reg_3 = _mm256_broadcast_ss((&alpha_));
  __m256 reg_4;
  reg_4 = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  __m256 reg_5;
  reg_5 = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  reg_5 = _mm256_fmadd_ps(reg_3, reg_4, reg_5);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg_5);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 reg_3;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_3 = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss((&alpha_)), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_4;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_4 = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 reg_5;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_5 = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), reg_3, _mm256_castsi256_ps(cmp));
    reg_5 = _mm256_fmadd_ps(prefixed_src1, reg_4, reg_5);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, reg_5);
    }
    
}
}

// exo_saxpy_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_saxpy_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  y.data[i * y.strides[0]] += alpha_ * x.data[i * x.strides[0]];
}
}


/* relying on the following instruction..."
mm256_broadcast_sd_scalar(out,val)
{out_data} = _mm256_broadcast_sd({val_data});
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
mm256_prefix_fmadd_pd(dst,src1,src2,bound)

{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {dst_data});
}}

*/

/* relying on the following instruction..."
mm256_prefix_fmadd_ps(dst,src1,src2,bound)

{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), {src1_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_fmadd_ps(prefixed_src1, {src2_data}, {dst_data});
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
