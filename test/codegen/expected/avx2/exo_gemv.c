#include "exo_gemv.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>




/* relying on the following instruction..."
avx2_assoc_reduce_add_pd_buffer(x,result)

    {{
        __m256d tmp = _mm256_hadd_pd({x_data}, {x_data});
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        {result_data} += _mm256_cvtsd_f64(tmp);
    }}
    
*/

/* relying on the following instruction..."
avx2_assoc_reduce_add_ps_buffer(x,result)

    {{
        __m256 tmp = _mm256_hadd_ps({x_data}, {x_data});
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        {result_data} += _mm256_cvtss_f32(tmp);
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
// exo_dgemv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  double *result = (double*) malloc(4 * sizeof(*result));
  result[0] = 0.0;
  result[1] = 0.0;
  result[2] = 0.0;
  result[3] = 0.0;
  __m256d var0;
  var0 = _mm256_setzero_pd();
  __m256d var5[4];
  var5[0] = _mm256_setzero_pd();
  var5[1] = _mm256_setzero_pd();
  var5[2] = _mm256_setzero_pd();
  var5[3] = _mm256_setzero_pd();
  for (int_fast32_t joo = 0; joo < ((((3 + n) / (4)) - 1) / (4)); joo++) {
    __m256d tmp[4];
    __m256d var1[4];
    __m256d var2[4];
    __m256d var3[4];
    __m256d var4[4];
    tmp[0] = _mm256_loadu_pd(&x.data[16 * joo]);
    tmp[1] = _mm256_loadu_pd(&x.data[4 + 16 * joo]);
    tmp[2] = _mm256_loadu_pd(&x.data[8 + 16 * joo]);
    tmp[3] = _mm256_loadu_pd(&x.data[12 + 16 * joo]);
    var1[0] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 16 * joo]);
    var1[1] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var1[2] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var1[3] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    var5[0] = _mm256_fmadd_pd(tmp[0], var1[0], var5[0]);
    var5[1] = _mm256_fmadd_pd(tmp[1], var1[1], var5[1]);
    var5[2] = _mm256_fmadd_pd(tmp[2], var1[2], var5[2]);
    var5[3] = _mm256_fmadd_pd(tmp[3], var1[3], var5[3]);
    var2[0] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var2[1] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var2[2] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var2[3] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    var5[0] = _mm256_fmadd_pd(tmp[0], var2[0], var5[0]);
    var5[1] = _mm256_fmadd_pd(tmp[1], var2[1], var5[1]);
    var5[2] = _mm256_fmadd_pd(tmp[2], var2[2], var5[2]);
    var5[3] = _mm256_fmadd_pd(tmp[3], var2[3], var5[3]);
    var3[0] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var3[1] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var3[2] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var3[3] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    var5[0] = _mm256_fmadd_pd(tmp[0], var3[0], var5[0]);
    var5[1] = _mm256_fmadd_pd(tmp[1], var3[1], var5[1]);
    var5[2] = _mm256_fmadd_pd(tmp[2], var3[2], var5[2]);
    var5[3] = _mm256_fmadd_pd(tmp[3], var3[3], var5[3]);
    var4[0] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var4[1] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var4[2] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var4[3] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    var5[0] = _mm256_fmadd_pd(tmp[0], var4[0], var5[0]);
    var5[1] = _mm256_fmadd_pd(tmp[1], var4[1], var5[1]);
    var5[2] = _mm256_fmadd_pd(tmp[2], var4[2], var5[2]);
    var5[3] = _mm256_fmadd_pd(tmp[3], var4[3], var5[3]);
  }
  var0 = _mm256_add_pd(var5[0], var0);
  var0 = _mm256_add_pd(var5[1], var0);
  var0 = _mm256_add_pd(var5[2], var0);
  var0 = _mm256_add_pd(var5[3], var0);
  for (int_fast32_t joi = 0; joi < (((3 + n) / (4)) - 1) % 4; joi++) {
    __m256d tmp;
    __m256d var1;
    __m256d var2;
    __m256d var3;
    __m256d var4;
    tmp = _mm256_loadu_pd(&x.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var1 = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var0 = _mm256_fmadd_pd(tmp, var1, var0);
    var2 = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var0 = _mm256_fmadd_pd(tmp, var2, var0);
    var3 = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var0 = _mm256_fmadd_pd(tmp, var3, var0);
    var4 = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var0 = _mm256_fmadd_pd(tmp, var4, var0);
  }
  for (int_fast32_t jo = ((3 + n) / (4)) - 1; jo < ((3 + n) / (4)); jo++) {
    __m256d tmp;
    __m256d var1;
    __m256d var2;
    __m256d var3;
    __m256d var4;
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            tmp = _mm256_maskload_pd(&x.data[4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1 = _mm256_maskload_pd(&A.data[(4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var1, var0);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var2 = _mm256_maskload_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var2, var0);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var3 = _mm256_maskload_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var3, var0);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var4 = _mm256_maskload_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var4, var0);
}

  }
  
    {
        __m256d tmp = _mm256_hadd_pd(var0, var0);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[0] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[4 * io] = beta_ * y.data[4 * io] + alpha_ * result[0];
  y.data[1 + 4 * io] = beta_ * y.data[1 + 4 * io] + alpha_ * result[1];
  y.data[2 + 4 * io] = beta_ * y.data[2 + 4 * io] + alpha_ * result[2];
  y.data[3 + 4 * io] = beta_ * y.data[3 + 4 * io] + alpha_ * result[3];
  free(result);
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
  y.data[ii + (m / 4) * 4] = beta_ * y.data[ii + (m / 4) * 4] + alpha_ * result;
}
}

// exo_dgemv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = beta_ * y.data[i * y.strides[0]] + alpha_ * result;
}
}

// exo_dgemv_row_major_Trans_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dgemv_row_major_Trans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k] = beta_ * y.data[k];
}
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  double *alphaXi = (double*) malloc(4 * sizeof(*alphaXi));
  alphaXi[0] = alpha_ * x.data[4 * io];
  alphaXi[1] = alpha_ * x.data[1 + 4 * io];
  alphaXi[2] = alpha_ * x.data[2 + 4 * io];
  alphaXi[3] = alpha_ * x.data[3 + 4 * io];
  __m256d var0;
  var0 = _mm256_broadcast_sd(&alphaXi[0]);
  __m256d var2;
  var2 = _mm256_broadcast_sd(&alphaXi[1]);
  __m256d var4;
  var4 = _mm256_broadcast_sd(&alphaXi[2]);
  __m256d var6;
  var6 = _mm256_broadcast_sd(&alphaXi[3]);
  free(alphaXi);
  for (int_fast32_t joo = 0; joo < ((((3 + n) / (4)) - 1) / (4)); joo++) {
    __m256d tmp[4];
    __m256d var1[4];
    __m256d var3[4];
    __m256d var5[4];
    __m256d var7[4];
    tmp[0] = _mm256_loadu_pd(&y.data[16 * joo]);
    tmp[1] = _mm256_loadu_pd(&y.data[4 + 16 * joo]);
    tmp[2] = _mm256_loadu_pd(&y.data[8 + 16 * joo]);
    tmp[3] = _mm256_loadu_pd(&y.data[12 + 16 * joo]);
    var1[0] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 16 * joo]);
    var1[1] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var1[2] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var1[3] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    tmp[0] = _mm256_fmadd_pd(var0, var1[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var0, var1[1], tmp[1]);
    tmp[2] = _mm256_fmadd_pd(var0, var1[2], tmp[2]);
    tmp[3] = _mm256_fmadd_pd(var0, var1[3], tmp[3]);
    var3[0] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var3[1] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var3[2] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var3[3] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    tmp[0] = _mm256_fmadd_pd(var2, var3[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var2, var3[1], tmp[1]);
    tmp[2] = _mm256_fmadd_pd(var2, var3[2], tmp[2]);
    tmp[3] = _mm256_fmadd_pd(var2, var3[3], tmp[3]);
    var5[0] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var5[1] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var5[2] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var5[3] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    tmp[0] = _mm256_fmadd_pd(var4, var5[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var4, var5[1], tmp[1]);
    tmp[2] = _mm256_fmadd_pd(var4, var5[2], tmp[2]);
    tmp[3] = _mm256_fmadd_pd(var4, var5[3], tmp[3]);
    var7[0] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var7[1] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 + 16 * joo]);
    var7[2] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var7[3] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 12 + 16 * joo]);
    tmp[0] = _mm256_fmadd_pd(var6, var7[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var6, var7[1], tmp[1]);
    tmp[2] = _mm256_fmadd_pd(var6, var7[2], tmp[2]);
    tmp[3] = _mm256_fmadd_pd(var6, var7[3], tmp[3]);
    _mm256_storeu_pd(&y.data[16 * joo], tmp[0]);
    _mm256_storeu_pd(&y.data[4 + 16 * joo], tmp[1]);
    _mm256_storeu_pd(&y.data[8 + 16 * joo], tmp[2]);
    _mm256_storeu_pd(&y.data[12 + 16 * joo], tmp[3]);
  }
  for (int_fast32_t joi = 0; joi < (((3 + n) / (4)) - 1) % 4; joi++) {
    __m256d tmp;
    __m256d var1;
    __m256d var3;
    __m256d var5;
    __m256d var7;
    tmp = _mm256_loadu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    var1 = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    tmp = _mm256_fmadd_pd(var0, var1, tmp);
    var3 = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    tmp = _mm256_fmadd_pd(var2, var3, tmp);
    var5 = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    tmp = _mm256_fmadd_pd(var4, var5, tmp);
    var7 = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi]);
    tmp = _mm256_fmadd_pd(var6, var7, tmp);
    _mm256_storeu_pd(&y.data[16 * ((((3 + n) / 4) - 1) / 4) + 4 * joi], tmp);
  }
  for (int_fast32_t jo = ((3 + n) / (4)) - 1; jo < ((3 + n) / (4)); jo++) {
    __m256d tmp;
    __m256d var1;
    __m256d var3;
    __m256d var5;
    __m256d var7;
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            tmp = _mm256_maskload_pd(&y.data[4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1 = _mm256_maskload_pd(&A.data[(4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var1, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var3 = _mm256_maskload_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var2, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var3, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var5 = _mm256_maskload_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var4, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var5, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var7 = _mm256_maskload_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var6, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var7, tmp);
}

    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * jo], cmp, tmp);
    }
    
  }
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 4) * 4];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
}
}

// exo_dgemv_row_major_Trans_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][m] @DRAM,
//     y : [f64][n] @DRAM
// )
void exo_dgemv_row_major_Trans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k * y.strides[0]] = beta_ * y.data[k * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[i * x.strides[0]];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j * y.strides[0]] += alphaXi * A.data[i * A.strides[0] + j];
  }
}
}

// exo_sgemv_row_major_NonTrans_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_row_major_NonTrans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  float *result = (float*) malloc(4 * sizeof(*result));
  result[0] = 0.0;
  result[1] = 0.0;
  result[2] = 0.0;
  result[3] = 0.0;
  __m256 var0;
  var0 = _mm256_setzero_ps();
  __m256 var5[4];
  var5[0] = _mm256_setzero_ps();
  var5[1] = _mm256_setzero_ps();
  var5[2] = _mm256_setzero_ps();
  var5[3] = _mm256_setzero_ps();
  for (int_fast32_t joo = 0; joo < ((((7 + n) / (8)) - 1) / (4)); joo++) {
    __m256 tmp[4];
    __m256 var1[4];
    __m256 var2[4];
    __m256 var3[4];
    __m256 var4[4];
    tmp[0] = _mm256_loadu_ps(&x.data[32 * joo]);
    tmp[1] = _mm256_loadu_ps(&x.data[8 + 32 * joo]);
    tmp[2] = _mm256_loadu_ps(&x.data[16 + 32 * joo]);
    tmp[3] = _mm256_loadu_ps(&x.data[24 + 32 * joo]);
    var1[0] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 32 * joo]);
    var1[1] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var1[2] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var1[3] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    var5[0] = _mm256_fmadd_ps(tmp[0], var1[0], var5[0]);
    var5[1] = _mm256_fmadd_ps(tmp[1], var1[1], var5[1]);
    var5[2] = _mm256_fmadd_ps(tmp[2], var1[2], var5[2]);
    var5[3] = _mm256_fmadd_ps(tmp[3], var1[3], var5[3]);
    var2[0] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var2[1] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var2[2] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var2[3] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    var5[0] = _mm256_fmadd_ps(tmp[0], var2[0], var5[0]);
    var5[1] = _mm256_fmadd_ps(tmp[1], var2[1], var5[1]);
    var5[2] = _mm256_fmadd_ps(tmp[2], var2[2], var5[2]);
    var5[3] = _mm256_fmadd_ps(tmp[3], var2[3], var5[3]);
    var3[0] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var3[1] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var3[2] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var3[3] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    var5[0] = _mm256_fmadd_ps(tmp[0], var3[0], var5[0]);
    var5[1] = _mm256_fmadd_ps(tmp[1], var3[1], var5[1]);
    var5[2] = _mm256_fmadd_ps(tmp[2], var3[2], var5[2]);
    var5[3] = _mm256_fmadd_ps(tmp[3], var3[3], var5[3]);
    var4[0] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var4[1] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var4[2] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var4[3] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    var5[0] = _mm256_fmadd_ps(tmp[0], var4[0], var5[0]);
    var5[1] = _mm256_fmadd_ps(tmp[1], var4[1], var5[1]);
    var5[2] = _mm256_fmadd_ps(tmp[2], var4[2], var5[2]);
    var5[3] = _mm256_fmadd_ps(tmp[3], var4[3], var5[3]);
  }
  var0 = _mm256_add_ps(var5[0], var0);
  var0 = _mm256_add_ps(var5[1], var0);
  var0 = _mm256_add_ps(var5[2], var0);
  var0 = _mm256_add_ps(var5[3], var0);
  for (int_fast32_t joi = 0; joi < (((7 + n) / (8)) - 1) % 4; joi++) {
    __m256 tmp;
    __m256 var1;
    __m256 var2;
    __m256 var3;
    __m256 var4;
    tmp = _mm256_loadu_ps(&x.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var1 = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var0 = _mm256_fmadd_ps(tmp, var1, var0);
    var2 = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var0 = _mm256_fmadd_ps(tmp, var2, var0);
    var3 = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var0 = _mm256_fmadd_ps(tmp, var3, var0);
    var4 = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var0 = _mm256_fmadd_ps(tmp, var4, var0);
  }
  for (int_fast32_t jo = ((7 + n) / (8)) - 1; jo < ((7 + n) / (8)); jo++) {
    __m256 tmp;
    __m256 var1;
    __m256 var2;
    __m256 var3;
    __m256 var4;
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    tmp = _mm256_maskload_ps(&x.data[8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1 = _mm256_maskload_ps(&A.data[(4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var1, var0);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var2 = _mm256_maskload_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var2, var0);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var3 = _mm256_maskload_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var3, var0);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var4 = _mm256_maskload_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var4, var0);
}

  }
  
    {
        __m256 tmp = _mm256_hadd_ps(var0, var0);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[0] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[4 * io] = beta_ * y.data[4 * io] + alpha_ * result[0];
  y.data[1 + 4 * io] = beta_ * y.data[1 + 4 * io] + alpha_ * result[1];
  y.data[2 + 4 * io] = beta_ * y.data[2 + 4 * io] + alpha_ * result[2];
  y.data[3 + 4 * io] = beta_ * y.data[3 + 4 * io] + alpha_ * result[3];
  free(result);
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
  y.data[ii + (m / 4) * 4] = beta_ * y.data[ii + (m / 4) * 4] + alpha_ * result;
}
}

// exo_sgemv_row_major_NonTrans_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_row_major_NonTrans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  y.data[i * y.strides[0]] = beta_ * y.data[i * y.strides[0]] + alpha_ * result;
}
}

// exo_sgemv_row_major_Trans_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sgemv_row_major_Trans_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k] = beta_ * y.data[k];
}
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  float *alphaXi = (float*) malloc(4 * sizeof(*alphaXi));
  alphaXi[0] = alpha_ * x.data[4 * io];
  alphaXi[1] = alpha_ * x.data[1 + 4 * io];
  alphaXi[2] = alpha_ * x.data[2 + 4 * io];
  alphaXi[3] = alpha_ * x.data[3 + 4 * io];
  __m256 var0;
  var0 = _mm256_broadcast_ss(&alphaXi[0]);
  __m256 var2;
  var2 = _mm256_broadcast_ss(&alphaXi[1]);
  __m256 var4;
  var4 = _mm256_broadcast_ss(&alphaXi[2]);
  __m256 var6;
  var6 = _mm256_broadcast_ss(&alphaXi[3]);
  free(alphaXi);
  for (int_fast32_t joo = 0; joo < ((((7 + n) / (8)) - 1) / (4)); joo++) {
    __m256 tmp[4];
    __m256 var1[4];
    __m256 var3[4];
    __m256 var5[4];
    __m256 var7[4];
    tmp[0] = _mm256_loadu_ps(&y.data[32 * joo]);
    tmp[1] = _mm256_loadu_ps(&y.data[8 + 32 * joo]);
    tmp[2] = _mm256_loadu_ps(&y.data[16 + 32 * joo]);
    tmp[3] = _mm256_loadu_ps(&y.data[24 + 32 * joo]);
    var1[0] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 32 * joo]);
    var1[1] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var1[2] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var1[3] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    tmp[0] = _mm256_fmadd_ps(var0, var1[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var0, var1[1], tmp[1]);
    tmp[2] = _mm256_fmadd_ps(var0, var1[2], tmp[2]);
    tmp[3] = _mm256_fmadd_ps(var0, var1[3], tmp[3]);
    var3[0] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var3[1] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var3[2] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var3[3] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    tmp[0] = _mm256_fmadd_ps(var2, var3[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var2, var3[1], tmp[1]);
    tmp[2] = _mm256_fmadd_ps(var2, var3[2], tmp[2]);
    tmp[3] = _mm256_fmadd_ps(var2, var3[3], tmp[3]);
    var5[0] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var5[1] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var5[2] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var5[3] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    tmp[0] = _mm256_fmadd_ps(var4, var5[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var4, var5[1], tmp[1]);
    tmp[2] = _mm256_fmadd_ps(var4, var5[2], tmp[2]);
    tmp[3] = _mm256_fmadd_ps(var4, var5[3], tmp[3]);
    var7[0] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 32 * joo]);
    var7[1] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 + 32 * joo]);
    var7[2] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 + 32 * joo]);
    var7[3] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 24 + 32 * joo]);
    tmp[0] = _mm256_fmadd_ps(var6, var7[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var6, var7[1], tmp[1]);
    tmp[2] = _mm256_fmadd_ps(var6, var7[2], tmp[2]);
    tmp[3] = _mm256_fmadd_ps(var6, var7[3], tmp[3]);
    _mm256_storeu_ps(&y.data[32 * joo], tmp[0]);
    _mm256_storeu_ps(&y.data[8 + 32 * joo], tmp[1]);
    _mm256_storeu_ps(&y.data[16 + 32 * joo], tmp[2]);
    _mm256_storeu_ps(&y.data[24 + 32 * joo], tmp[3]);
  }
  for (int_fast32_t joi = 0; joi < (((7 + n) / (8)) - 1) % 4; joi++) {
    __m256 tmp;
    __m256 var1;
    __m256 var3;
    __m256 var5;
    __m256 var7;
    tmp = _mm256_loadu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    var1 = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    tmp = _mm256_fmadd_ps(var0, var1, tmp);
    var3 = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    tmp = _mm256_fmadd_ps(var2, var3, tmp);
    var5 = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    tmp = _mm256_fmadd_ps(var4, var5, tmp);
    var7 = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi]);
    tmp = _mm256_fmadd_ps(var6, var7, tmp);
    _mm256_storeu_ps(&y.data[32 * ((((7 + n) / 8) - 1) / 4) + 8 * joi], tmp);
  }
  for (int_fast32_t jo = ((7 + n) / (8)) - 1; jo < ((7 + n) / (8)); jo++) {
    __m256 tmp;
    __m256 var1;
    __m256 var3;
    __m256 var5;
    __m256 var7;
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    tmp = _mm256_maskload_ps(&y.data[8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1 = _mm256_maskload_ps(&A.data[(4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var1, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var3 = _mm256_maskload_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var2, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var3, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var5 = _mm256_maskload_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var4, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var5, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var7 = _mm256_maskload_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var6, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var7, tmp);
}

    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * jo], cmp, tmp);
    }
    
  }
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 4) * 4];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
}
}

// exo_sgemv_row_major_Trans_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][m] @DRAM,
//     y : [f32][n] @DRAM
// )
void exo_sgemv_row_major_Trans_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t k = 0; k < n; k++) {
  y.data[k * y.strides[0]] = beta_ * y.data[k * y.strides[0]];
}
for (int_fast32_t i = 0; i < m; i++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[i * x.strides[0]];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j * y.strides[0]] += alphaXi * A.data[i * A.strides[0] + j];
  }
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
