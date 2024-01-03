#include "exo_swap.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>



// exo_dswap_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dswap_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t ioo = 0; ioo < ((((3 + n) / (4)) - 1) / (4)); ioo++) {
  __m256d tmp[4];
  __m256d reg[4];
  tmp[0] = _mm256_loadu_pd(&x.data[16 * ioo]);
  tmp[1] = _mm256_loadu_pd(&x.data[4 + 16 * ioo]);
  tmp[2] = _mm256_loadu_pd(&x.data[8 + 16 * ioo]);
  tmp[3] = _mm256_loadu_pd(&x.data[12 + 16 * ioo]);
  reg[0] = _mm256_loadu_pd(&y.data[16 * ioo]);
  reg[1] = _mm256_loadu_pd(&y.data[4 + 16 * ioo]);
  reg[2] = _mm256_loadu_pd(&y.data[8 + 16 * ioo]);
  reg[3] = _mm256_loadu_pd(&y.data[12 + 16 * ioo]);
  _mm256_storeu_pd(&x.data[16 * ioo], reg[0]);
  _mm256_storeu_pd(&x.data[4 + 16 * ioo], reg[1]);
  _mm256_storeu_pd(&x.data[8 + 16 * ioo], reg[2]);
  _mm256_storeu_pd(&x.data[12 + 16 * ioo], reg[3]);
  _mm256_storeu_pd(&y.data[16 * ioo], tmp[0]);
  _mm256_storeu_pd(&y.data[4 + 16 * ioo], tmp[1]);
  _mm256_storeu_pd(&y.data[8 + 16 * ioo], tmp[2]);
  _mm256_storeu_pd(&y.data[12 + 16 * ioo], tmp[3]);
}
for (int_fast32_t ioi = 0; ioi < (((3 + n) / (4)) - 1) % 4; ioi++) {
  __m256d tmp;
  tmp = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  __m256d reg;
  reg = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi]);
  _mm256_storeu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], reg);
  _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * ioi], tmp);
}
for (int_fast32_t io = ((3 + n) / (4)) - 1; io < ((3 + n) / (4)); io++) {
  __m256d tmp;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            tmp = _mm256_maskload_pd(&x.data[4 * io], cmp);
       }
       
  __m256d reg;
  
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            reg = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&x.data[4 * io], cmp, reg);
    }
    
  
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, tmp);
    }
    
}
}

// exo_dswap_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dswap_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64 x, struct exo_win_1f64 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  double tmp;
  tmp = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = tmp;
}
}

// exo_sswap_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sswap_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y ) {
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
for (int_fast32_t ioo = 0; ioo < ((((7 + n) / (8)) - 1) / (4)); ioo++) {
  __m256 tmp[4];
  __m256 reg[4];
  tmp[0] = _mm256_loadu_ps(&x.data[32 * ioo]);
  tmp[1] = _mm256_loadu_ps(&x.data[8 + 32 * ioo]);
  tmp[2] = _mm256_loadu_ps(&x.data[16 + 32 * ioo]);
  tmp[3] = _mm256_loadu_ps(&x.data[24 + 32 * ioo]);
  reg[0] = _mm256_loadu_ps(&y.data[32 * ioo]);
  reg[1] = _mm256_loadu_ps(&y.data[8 + 32 * ioo]);
  reg[2] = _mm256_loadu_ps(&y.data[16 + 32 * ioo]);
  reg[3] = _mm256_loadu_ps(&y.data[24 + 32 * ioo]);
  _mm256_storeu_ps(&x.data[32 * ioo], reg[0]);
  _mm256_storeu_ps(&x.data[8 + 32 * ioo], reg[1]);
  _mm256_storeu_ps(&x.data[16 + 32 * ioo], reg[2]);
  _mm256_storeu_ps(&x.data[24 + 32 * ioo], reg[3]);
  _mm256_storeu_ps(&y.data[32 * ioo], tmp[0]);
  _mm256_storeu_ps(&y.data[8 + 32 * ioo], tmp[1]);
  _mm256_storeu_ps(&y.data[16 + 32 * ioo], tmp[2]);
  _mm256_storeu_ps(&y.data[24 + 32 * ioo], tmp[3]);
}
for (int_fast32_t ioi = 0; ioi < (((7 + n) / (8)) - 1) % 4; ioi++) {
  __m256 tmp;
  tmp = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  __m256 reg;
  reg = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi]);
  _mm256_storeu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], reg);
  _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * ioi], tmp);
}
for (int_fast32_t io = ((7 + n) / (8)) - 1; io < ((7 + n) / (8)); io++) {
  __m256 tmp;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    tmp = _mm256_maskload_ps(&x.data[8 * io], cmp);
}

  __m256 reg;
  
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    reg = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&x.data[8 * io], cmp, reg);
    }
    
  
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, tmp);
    }
    
}
}

// exo_sswap_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sswap_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32 x, struct exo_win_1f32 y ) {
for (int_fast32_t i = 0; i < n; i++) {
  float tmp;
  tmp = x.data[i * x.strides[0]];
  x.data[i * x.strides[0]] = y.data[i * y.strides[0]];
  y.data[i * y.strides[0]] = tmp;
}
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
