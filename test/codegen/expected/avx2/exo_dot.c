#include "exo_dot.h"



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
avx2_assoc_reduce_add_ps(x,result)

    {{
        __m256 tmp = _mm256_hadd_ps({x_data}, {x_data});
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *{result} += _mm256_cvtss_f32(tmp);
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
// exo_ddot_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_ddot_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64c y, double* result ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double result_;
result_ = 0.0;
__m256d var0;
var0 = _mm256_setzero_pd();
__m256d var3[4];
var3[0] = _mm256_setzero_pd();
var3[1] = _mm256_setzero_pd();
var3[2] = _mm256_setzero_pd();
var3[3] = _mm256_setzero_pd();
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d var1[4];
  __m256d var2[4];
  var1[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  var1[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  var1[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  var1[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  var2[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  var2[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  var2[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  var2[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  var3[0] = _mm256_fmadd_pd(var1[0], var2[0], var3[0]);
  var3[1] = _mm256_fmadd_pd(var1[1], var2[1], var3[1]);
  var3[2] = _mm256_fmadd_pd(var1[2], var2[2], var3[2]);
  var3[3] = _mm256_fmadd_pd(var1[3], var2[3], var3[3]);
}
var0 = _mm256_add_pd(var3[0], var0);
var0 = _mm256_add_pd(var3[1], var0);
var0 = _mm256_add_pd(var3[2], var0);
var0 = _mm256_add_pd(var3[3], var0);
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d var1;
  __m256d var2;
  var1 = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  var2 = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  var0 = _mm256_fmadd_pd(var1, var2, var0);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d var1;
  __m256d var2;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1 = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var2 = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var1, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var2, var0);
}

}

    {
        __m256d tmp = _mm256_hadd_pd(var0, var0);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&result_) += _mm256_cvtsd_f64(tmp);
    }
    
*result = result_;
}

// exo_ddot_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_ddot_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, struct exo_win_1f64c y, double* result ) {
double result_;
result_ = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  result_ += x.data[i * x.strides[0]] * y.data[i * y.strides[0]];
}
*result = result_;
}

// exo_sdot_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sdot_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32c y, float* result ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float result_;
result_ = 0.0;
__m256 var0;
var0 = _mm256_setzero_ps();
__m256 var3[4];
var3[0] = _mm256_setzero_ps();
var3[1] = _mm256_setzero_ps();
var3[2] = _mm256_setzero_ps();
var3[3] = _mm256_setzero_ps();
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 var1[4];
  __m256 var2[4];
  var1[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  var1[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  var1[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  var1[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  var2[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  var2[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  var2[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  var2[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  var3[0] = _mm256_fmadd_ps(var1[0], var2[0], var3[0]);
  var3[1] = _mm256_fmadd_ps(var1[1], var2[1], var3[1]);
  var3[2] = _mm256_fmadd_ps(var1[2], var2[2], var3[2]);
  var3[3] = _mm256_fmadd_ps(var1[3], var2[3], var3[3]);
}
var0 = _mm256_add_ps(var3[0], var0);
var0 = _mm256_add_ps(var3[1], var0);
var0 = _mm256_add_ps(var3[2], var0);
var0 = _mm256_add_ps(var3[3], var0);
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 var1;
  __m256 var2;
  var1 = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  var2 = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  var0 = _mm256_fmadd_ps(var1, var2, var0);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 var1;
  __m256 var2;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1 = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var2 = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var1, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var2, var0);
}

}

    {
        __m256 tmp = _mm256_hadd_ps(var0, var0);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(&result_) += _mm256_cvtss_f32(tmp);
    }
    
*result = result_;
}

// exo_sdot_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sdot_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, struct exo_win_1f32c y, float* result ) {
float result_;
result_ = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  result_ += x.data[i * x.strides[0]] * y.data[i * y.strides[0]];
}
*result = result_;
}


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
mm256_prefix_fmadd_reduce_ps(dst,src1,src2,bound)

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
mm256_setzero_pd(dst)
{dst_data} = _mm256_setzero_pd();
*/

/* relying on the following instruction..."
mm256_setzero_ps(dst)
{dst_data} = _mm256_setzero_ps();
*/
