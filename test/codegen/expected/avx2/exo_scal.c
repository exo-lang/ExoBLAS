#include "exo_scal.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>



// exo_dscal_alpha_0_stride_1(
//     n : size,
//     x : [f64][n] @DRAM
// )
void exo_dscal_alpha_0_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x ) {
// assert stride(x, 0) == 1
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d reg[4];
  reg[0] = _mm256_setzero_pd();
  reg[1] = _mm256_setzero_pd();
  reg[2] = _mm256_setzero_pd();
  reg[3] = _mm256_setzero_pd();
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    _mm256_storeu_pd(&x.data[16 * ioo], reg[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], reg[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], reg[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], reg[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d reg;
  reg = _mm256_setzero_pd();
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg);
  }
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d reg;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
reg = _mm256_blendv_pd (reg, _mm256_setzero_pd(), _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg);
    }
    
}
}

// exo_dscal_alpha_0_stride_any(
//     n : size,
//     x : [f64][n] @DRAM
// )
void exo_dscal_alpha_0_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x ) {
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = 0.0;
}
}

// exo_dscal_stride_1(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM
// )
void exo_dscal_stride_1( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64 x ) {
// assert stride(x, 0) == 1
double alpha_;
alpha_ = *alpha;
if (0 < ((((3 + n) / (4)) - 1) / (4))) {
  __m256d reg[4];
  __m256d reg_1[4];
  __m256d reg_2[4];
  reg[0] = _mm256_broadcast_sd((&alpha_));
  reg[1] = _mm256_broadcast_sd((&alpha_));
  reg[2] = _mm256_broadcast_sd((&alpha_));
  reg[3] = _mm256_broadcast_sd((&alpha_));
  for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
    reg_1[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
    reg_1[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
    reg_1[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
    reg_1[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
    reg_2[0] = _mm256_mul_pd(reg[0], reg_1[0]);
    reg_2[1] = _mm256_mul_pd(reg[1], reg_1[1]);
    reg_2[2] = _mm256_mul_pd(reg[2], reg_1[2]);
    reg_2[3] = _mm256_mul_pd(reg[3], reg_1[3]);
    _mm256_storeu_pd(&x.data[16 * ioo], reg_2[0]);
    _mm256_storeu_pd(&x.data[4 + 16 * ioo], reg_2[1]);
    _mm256_storeu_pd(&x.data[8 + 16 * ioo], reg_2[2]);
    _mm256_storeu_pd(&x.data[12 + 16 * ioo], reg_2[3]);
  }
}
if (0 < (((3 + n) / (4)) - 1) % 4) {
  __m256d reg;
  reg = _mm256_broadcast_sd((&alpha_));
  __m256d reg_1;
  __m256d reg_2;
  for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
    reg_1 = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
    reg_2 = _mm256_mul_pd(reg, reg_1);
    _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg_2);
  }
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d reg;
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    reg = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd((&alpha_)), _mm256_castsi256_pd(cmp));
    }
    
  __m256d reg_1;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            reg_1 = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d reg_2;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd(reg, reg_1);
reg_2 = _mm256_blendv_pd (reg_2, mul, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg_2);
    }
    
}
}

// exo_dscal_stride_any(
//     n : size,
//     alpha : f64 @DRAM,
//     x : [f64][n] @DRAM
// )
void exo_dscal_stride_any( void *ctxt, int_fast32_t n, const double* alpha, struct exo_win_1f64 x ) {
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = alpha_ * x.data[i * x.strides[0]];
}
}

// exo_sscal_alpha_0_stride_1(
//     n : size,
//     x : [f32][n] @DRAM
// )
void exo_sscal_alpha_0_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x ) {
// assert stride(x, 0) == 1
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 reg[4];
  reg[0] = _mm256_setzero_ps();
  reg[1] = _mm256_setzero_ps();
  reg[2] = _mm256_setzero_ps();
  reg[3] = _mm256_setzero_ps();
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    _mm256_storeu_ps(&x.data[32 * ioo], reg[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], reg[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], reg[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], reg[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 reg;
  reg = _mm256_setzero_ps();
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg);
  }
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 reg;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
reg = _mm256_blendv_ps (reg, _mm256_setzero_ps(), _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg);
    }
    
}
}

// exo_sscal_alpha_0_stride_any(
//     n : size,
//     x : [f32][n] @DRAM
// )
void exo_sscal_alpha_0_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x ) {
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = 0.0;
}
}

// exo_sscal_stride_1(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM
// )
void exo_sscal_stride_1( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32 x ) {
// assert stride(x, 0) == 1
float alpha_;
alpha_ = *alpha;
if (0 < ((((7 + n) / (8)) - 1) / (4))) {
  __m256 reg[4];
  __m256 reg_1[4];
  __m256 reg_2[4];
  reg[0] = _mm256_broadcast_ss((&alpha_));
  reg[1] = _mm256_broadcast_ss((&alpha_));
  reg[2] = _mm256_broadcast_ss((&alpha_));
  reg[3] = _mm256_broadcast_ss((&alpha_));
  for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
    reg_1[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
    reg_1[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
    reg_1[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
    reg_1[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
    reg_2[0] = _mm256_mul_ps(reg[0], reg_1[0]);
    reg_2[1] = _mm256_mul_ps(reg[1], reg_1[1]);
    reg_2[2] = _mm256_mul_ps(reg[2], reg_1[2]);
    reg_2[3] = _mm256_mul_ps(reg[3], reg_1[3]);
    _mm256_storeu_ps(&x.data[32 * ioo], reg_2[0]);
    _mm256_storeu_ps(&x.data[8 + 32 * ioo], reg_2[1]);
    _mm256_storeu_ps(&x.data[16 + 32 * ioo], reg_2[2]);
    _mm256_storeu_ps(&x.data[24 + 32 * ioo], reg_2[3]);
  }
}
if (0 < (((7 + n) / (8)) - 1) % 4) {
  __m256 reg;
  reg = _mm256_broadcast_ss((&alpha_));
  __m256 reg_1;
  __m256 reg_2;
  for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
    reg_1 = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
    reg_2 = _mm256_mul_ps(reg, reg_1);
    _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg_2);
  }
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 reg;
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss((&alpha_)), _mm256_castsi256_ps(cmp));
    }
    
  __m256 reg_1;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg_1 = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 reg_2;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(reg, reg_1);
reg_2 = _mm256_blendv_ps (reg_2, mul, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg_2);
    }
    
}
}

// exo_sscal_stride_any(
//     n : size,
//     alpha : f32 @DRAM,
//     x : [f32][n] @DRAM
// )
void exo_sscal_stride_any( void *ctxt, int_fast32_t n, const float* alpha, struct exo_win_1f32 x ) {
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < n; i++) {
  x.data[i * x.strides[0]] = alpha_ * x.data[i * x.strides[0]];
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
mm256_prefix_setzero_pd(dst,bound)

{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
{dst_data} = _mm256_blendv_pd ({dst_data}, _mm256_setzero_pd(), _mm256_castsi256_pd(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_setzero_ps(dst,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
{dst_data} = _mm256_blendv_ps ({dst_data}, _mm256_setzero_ps(), _mm256_castsi256_ps(cmp));
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
mm256_setzero_pd(dst)
{dst_data} = _mm256_setzero_pd();
*/

/* relying on the following instruction..."
mm256_setzero_ps(dst)
{dst_data} = _mm256_setzero_ps();
*/

/* relying on the following instruction..."
mm256_storeu_pd(dst,src)
_mm256_storeu_pd(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
