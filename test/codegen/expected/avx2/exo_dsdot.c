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
avx2_fused_load_cvtps_pd(dst,src)
{dst_data} = _mm256_cvtps_pd(_mm_loadu_ps(&{src_data}));
*/

/* relying on the following instruction..."
avx2_prefix_fused_load_cvtps_pd(dst,src,bound)

{{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32({bound});
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_cvtps_pd(_mm_maskload_ps(&{src_data}, cmp));
}}

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
__m256d var0;
var0 = _mm256_setzero_pd();
__m256d var1[4];
var1[0] = _mm256_setzero_pd();
var1[1] = _mm256_setzero_pd();
var1[2] = _mm256_setzero_pd();
var1[3] = _mm256_setzero_pd();
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d d_x[4];
  __m256d d_y[4];
  d_x[0] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[16 * ioo]));
  d_x[1] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[4 + 16 * ioo]));
  d_x[2] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[8 + 16 * ioo]));
  d_x[3] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[12 + 16 * ioo]));
  d_y[0] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[16 * ioo]));
  d_y[1] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[4 + 16 * ioo]));
  d_y[2] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[8 + 16 * ioo]));
  d_y[3] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[12 + 16 * ioo]));
  var1[0] = _mm256_fmadd_pd(d_x[0], d_y[0], var1[0]);
  var1[1] = _mm256_fmadd_pd(d_x[1], d_y[1], var1[1]);
  var1[2] = _mm256_fmadd_pd(d_x[2], d_y[2], var1[2]);
  var1[3] = _mm256_fmadd_pd(d_x[3], d_y[3], var1[3]);
}
var0 = _mm256_add_pd(var1[0], var0);
var0 = _mm256_add_pd(var1[1], var0);
var0 = _mm256_add_pd(var1[2], var0);
var0 = _mm256_add_pd(var1[3], var0);
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d d_x;
  __m256d d_y;
  d_x = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]));
  d_y = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]));
  var0 = _mm256_fmadd_pd(d_x, d_y, var0);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d d_x;
  __m256d d_y;
  
{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32((-(4 * io) + n));
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    d_x = _mm256_cvtps_pd(_mm_maskload_ps(&x.data[4 * io], cmp));
}

  
{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32((-(4 * io) + n));
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    d_y = _mm256_cvtps_pd(_mm_maskload_ps(&y.data[4 * io], cmp));
}

  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), d_x, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, d_y, var0);
}

}

    {
        __m256d tmp = _mm256_hadd_pd(var0, var0);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&d_dot) += _mm256_cvtsd_f64(tmp);
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
__m256d var0;
var0 = _mm256_setzero_pd();
__m256d var1[4];
var1[0] = _mm256_setzero_pd();
var1[1] = _mm256_setzero_pd();
var1[2] = _mm256_setzero_pd();
var1[3] = _mm256_setzero_pd();
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d d_x[4];
  __m256d d_y[4];
  d_x[0] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[16 * ioo]));
  d_x[1] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[4 + 16 * ioo]));
  d_x[2] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[8 + 16 * ioo]));
  d_x[3] = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[12 + 16 * ioo]));
  d_y[0] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[16 * ioo]));
  d_y[1] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[4 + 16 * ioo]));
  d_y[2] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[8 + 16 * ioo]));
  d_y[3] = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[12 + 16 * ioo]));
  var1[0] = _mm256_fmadd_pd(d_x[0], d_y[0], var1[0]);
  var1[1] = _mm256_fmadd_pd(d_x[1], d_y[1], var1[1]);
  var1[2] = _mm256_fmadd_pd(d_x[2], d_y[2], var1[2]);
  var1[3] = _mm256_fmadd_pd(d_x[3], d_y[3], var1[3]);
}
var0 = _mm256_add_pd(var1[0], var0);
var0 = _mm256_add_pd(var1[1], var0);
var0 = _mm256_add_pd(var1[2], var0);
var0 = _mm256_add_pd(var1[3], var0);
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d d_x;
  __m256d d_y;
  d_x = _mm256_cvtps_pd(_mm_loadu_ps(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]));
  d_y = _mm256_cvtps_pd(_mm_loadu_ps(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]));
  var0 = _mm256_fmadd_pd(d_x, d_y, var0);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d d_x;
  __m256d d_y;
  
{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32((-(4 * io) + n));
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    d_x = _mm256_cvtps_pd(_mm_maskload_ps(&x.data[4 * io], cmp));
}

  
{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32((-(4 * io) + n));
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    d_y = _mm256_cvtps_pd(_mm_maskload_ps(&y.data[4 * io], cmp));
}

  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), d_x, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, d_y, var0);
}

}

    {
        __m256d tmp = _mm256_hadd_pd(var0, var0);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&d_dot) += _mm256_cvtsd_f64(tmp);
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
mm256_fmadd_reduce_pd(dst,src1,src2)
{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_prefix_fmadd_reduce_pd(dst,src1,src2,bound)

{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {dst_data});
}}

*/

/* relying on the following instruction..."
mm256_setzero_pd(dst)
{dst_data} = _mm256_setzero_pd();
*/
