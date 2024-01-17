#include "exo_asum.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

double _select_(double x, double v, double y, double z) {
    if (x < v) return y;
    else return z;
}



/* relying on the following instruction..."
avx2_abs_pd(dst,src)

{dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

*/

/* relying on the following instruction..."
avx2_abs_ps(dst,src)
{dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
*/

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
avx2_prefix_abs_pd(dst,src,bound)

{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_abs = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
{dst_data} = _mm256_blendv_pd ({dst_data}, src_abs, _mm256_castsi256_pd(cmp));
}}

*/

/* relying on the following instruction..."
avx2_prefix_abs_ps(dst,src,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_abs = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
{dst_data} = _mm256_blendv_ps ({dst_data}, src_abs, _mm256_castsi256_ps(cmp));
}}

*/

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
// exo_dasum_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dasum_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result ) {
// assert stride(x, 0) == 1
double result_;
result_ = 0.0;
__m256d var0;
var0 = _mm256_setzero_pd();
__m256d var1[7];
var1[0] = _mm256_setzero_pd();
var1[1] = _mm256_setzero_pd();
var1[2] = _mm256_setzero_pd();
var1[3] = _mm256_setzero_pd();
var1[4] = _mm256_setzero_pd();
var1[5] = _mm256_setzero_pd();
var1[6] = _mm256_setzero_pd();
for (int_fast32_t iooo = 0; iooo < ((((3 + n) / (4)) - 1) / (7)); iooo++) {
  __m256d xReg[1][7];
  __m256d selectReg[1][7];
  xReg[0][0] = _mm256_loadu_pd(&x.data[28 * iooo]);
  xReg[0][1] = _mm256_loadu_pd(&x.data[4 + 28 * iooo]);
  xReg[0][2] = _mm256_loadu_pd(&x.data[8 + 28 * iooo]);
  xReg[0][3] = _mm256_loadu_pd(&x.data[12 + 28 * iooo]);
  xReg[0][4] = _mm256_loadu_pd(&x.data[16 + 28 * iooo]);
  xReg[0][5] = _mm256_loadu_pd(&x.data[20 + 28 * iooo]);
  xReg[0][6] = _mm256_loadu_pd(&x.data[24 + 28 * iooo]);
  
selectReg[0][0] = _mm256_and_pd(xReg[0][0], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][1] = _mm256_and_pd(xReg[0][1], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][2] = _mm256_and_pd(xReg[0][2], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][3] = _mm256_and_pd(xReg[0][3], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][4] = _mm256_and_pd(xReg[0][4], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][5] = _mm256_and_pd(xReg[0][5], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  
selectReg[0][6] = _mm256_and_pd(xReg[0][6], _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  var1[0] = _mm256_add_pd(selectReg[0][0], var1[0]);
  var1[1] = _mm256_add_pd(selectReg[0][1], var1[1]);
  var1[2] = _mm256_add_pd(selectReg[0][2], var1[2]);
  var1[3] = _mm256_add_pd(selectReg[0][3], var1[3]);
  var1[4] = _mm256_add_pd(selectReg[0][4], var1[4]);
  var1[5] = _mm256_add_pd(selectReg[0][5], var1[5]);
  var1[6] = _mm256_add_pd(selectReg[0][6], var1[6]);
}
var0 = _mm256_add_pd(var1[0], var0);
var0 = _mm256_add_pd(var1[1], var0);
var0 = _mm256_add_pd(var1[2], var0);
var0 = _mm256_add_pd(var1[3], var0);
var0 = _mm256_add_pd(var1[4], var0);
var0 = _mm256_add_pd(var1[5], var0);
var0 = _mm256_add_pd(var1[6], var0);
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 7; ioi++) {
  __m256d xReg;
  xReg = _mm256_loadu_pd(&x.data[28 * ((((3 + n) / 4) - 1) / 7) + 4 * ioi]);
  __m256d selectReg;
  
selectReg = _mm256_and_pd(xReg, _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));

  var0 = _mm256_add_pd(selectReg, var0);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d xRegTail;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            xRegTail = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d selectRegTail;
  
{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_abs = _mm256_and_pd(xRegTail, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
selectRegTail = _mm256_blendv_pd (selectRegTail, src_abs, _mm256_castsi256_pd(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src = _mm256_blendv_pd (_mm256_setzero_pd(), selectRegTail, _mm256_castsi256_pd(cmp));
    var0 = _mm256_add_pd(prefixed_src, var0);
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

// exo_dasum_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dasum_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result ) {
double result_;
result_ = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  double arg;
  arg = 0.0;
  double arg_1;
  arg_1 = x.data[i * x.strides[0]];
  double arg_2;
  arg_2 = x.data[i * x.strides[0]];
  double arg_3;
  arg_3 = -x.data[i * x.strides[0]];
  result_ += (double)(_select_((double)*&arg, (double)*&arg_1, (double)*&arg_2, (double)*&arg_3));
}
*result = result_;
}

// exo_sasum_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sasum_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result ) {
// assert stride(x, 0) == 1
float result_;
result_ = 0.0;
__m256 var0;
var0 = _mm256_setzero_ps();
__m256 var1[7];
var1[0] = _mm256_setzero_ps();
var1[1] = _mm256_setzero_ps();
var1[2] = _mm256_setzero_ps();
var1[3] = _mm256_setzero_ps();
var1[4] = _mm256_setzero_ps();
var1[5] = _mm256_setzero_ps();
var1[6] = _mm256_setzero_ps();
for (int_fast32_t iooo = 0; iooo < ((((7 + n) / (8)) - 1) / (7)); iooo++) {
  __m256 xReg[1][7];
  __m256 selectReg[1][7];
  xReg[0][0] = _mm256_loadu_ps(&x.data[56 * iooo]);
  xReg[0][1] = _mm256_loadu_ps(&x.data[8 + 56 * iooo]);
  xReg[0][2] = _mm256_loadu_ps(&x.data[16 + 56 * iooo]);
  xReg[0][3] = _mm256_loadu_ps(&x.data[24 + 56 * iooo]);
  xReg[0][4] = _mm256_loadu_ps(&x.data[32 + 56 * iooo]);
  xReg[0][5] = _mm256_loadu_ps(&x.data[40 + 56 * iooo]);
  xReg[0][6] = _mm256_loadu_ps(&x.data[48 + 56 * iooo]);
  selectReg[0][0] = _mm256_and_ps(xReg[0][0], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][1] = _mm256_and_ps(xReg[0][1], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][2] = _mm256_and_ps(xReg[0][2], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][3] = _mm256_and_ps(xReg[0][3], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][4] = _mm256_and_ps(xReg[0][4], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][5] = _mm256_and_ps(xReg[0][5], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  selectReg[0][6] = _mm256_and_ps(xReg[0][6], _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  var1[0] = _mm256_add_ps(selectReg[0][0], var1[0]);
  var1[1] = _mm256_add_ps(selectReg[0][1], var1[1]);
  var1[2] = _mm256_add_ps(selectReg[0][2], var1[2]);
  var1[3] = _mm256_add_ps(selectReg[0][3], var1[3]);
  var1[4] = _mm256_add_ps(selectReg[0][4], var1[4]);
  var1[5] = _mm256_add_ps(selectReg[0][5], var1[5]);
  var1[6] = _mm256_add_ps(selectReg[0][6], var1[6]);
}
var0 = _mm256_add_ps(var1[0], var0);
var0 = _mm256_add_ps(var1[1], var0);
var0 = _mm256_add_ps(var1[2], var0);
var0 = _mm256_add_ps(var1[3], var0);
var0 = _mm256_add_ps(var1[4], var0);
var0 = _mm256_add_ps(var1[5], var0);
var0 = _mm256_add_ps(var1[6], var0);
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 7; ioi++) {
  __m256 xReg;
  xReg = _mm256_loadu_ps(&x.data[56 * ((((7 + n) / 8) - 1) / 7) + 8 * ioi]);
  __m256 selectReg;
  selectReg = _mm256_and_ps(xReg, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
  var0 = _mm256_add_ps(selectReg, var0);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 xRegTail;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    xRegTail = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 selectRegTail;
  
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_abs = _mm256_and_ps(xRegTail, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
selectRegTail = _mm256_blendv_ps (selectRegTail, src_abs, _mm256_castsi256_ps(cmp));
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src = _mm256_blendv_ps (_mm256_setzero_ps(), selectRegTail, _mm256_castsi256_ps(cmp));
    var0 = _mm256_add_ps(prefixed_src, var0);
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

// exo_sasum_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_sasum_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result ) {
float result_;
result_ = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  float arg;
  arg = 0.0;
  float arg_1;
  arg_1 = x.data[i * x.strides[0]];
  float arg_2;
  arg_2 = x.data[i * x.strides[0]];
  float arg_3;
  arg_3 = -x.data[i * x.strides[0]];
  result_ += _select_((double)*&arg, (double)*&arg_1, (double)*&arg_2, (double)*&arg_3);
}
*result = result_;
}


/* relying on the following instruction..."
mm256_loadu_pd(dst,src)
{dst_data} = _mm256_loadu_pd(&{src_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
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
