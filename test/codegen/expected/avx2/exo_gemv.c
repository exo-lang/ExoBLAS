#include "exo_gemv.h"



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
// exo_dgemv_rm_nt_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_rm_nt_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  double result_0;
  double result_1;
  double result_2;
  double result_3;
  y.data[4 * io] = y.data[4 * io] * beta_;
  result_0 = 0.0;
  y.data[1 + 4 * io] = y.data[1 + 4 * io] * beta_;
  result_1 = 0.0;
  y.data[2 + 4 * io] = y.data[2 + 4 * io] * beta_;
  result_2 = 0.0;
  y.data[3 + 4 * io] = y.data[3 + 4 * io] * beta_;
  result_3 = 0.0;
  __m256d var0;
  var0 = _mm256_setzero_pd();
  __m256d var1;
  var1 = _mm256_setzero_pd();
  __m256d var2;
  var2 = _mm256_setzero_pd();
  __m256d var3;
  var3 = _mm256_setzero_pd();
  __m256d var8[2];
  var8[0] = _mm256_setzero_pd();
  var8[1] = _mm256_setzero_pd();
  __m256d var9[2];
  var9[0] = _mm256_setzero_pd();
  var9[1] = _mm256_setzero_pd();
  __m256d var10[2];
  var10[0] = _mm256_setzero_pd();
  var10[1] = _mm256_setzero_pd();
  __m256d var11[2];
  var11[0] = _mm256_setzero_pd();
  var11[1] = _mm256_setzero_pd();
  for (int_fast32_t joo = 0; joo < ((((3 + n) / (4)) - 1) / (2)); joo++) {
    __m256d tmp[2];
    __m256d var4[2];
    __m256d var5[2];
    __m256d var6[2];
    __m256d var7[2];
    tmp[0] = _mm256_loadu_pd(&x.data[8 * joo]);
    tmp[1] = _mm256_loadu_pd(&x.data[4 + 8 * joo]);
    var4[0] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 * joo]);
    var4[1] = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var8[0] = _mm256_fmadd_pd(tmp[0], var4[0], var8[0]);
    var8[1] = _mm256_fmadd_pd(tmp[1], var4[1], var8[1]);
    var5[0] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var5[1] = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var9[0] = _mm256_fmadd_pd(tmp[0], var5[0], var9[0]);
    var9[1] = _mm256_fmadd_pd(tmp[1], var5[1], var9[1]);
    var6[0] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var6[1] = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var10[0] = _mm256_fmadd_pd(tmp[0], var6[0], var10[0]);
    var10[1] = _mm256_fmadd_pd(tmp[1], var6[1], var10[1]);
    var7[0] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * joo]);
    var7[1] = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 + 8 * joo]);
    var11[0] = _mm256_fmadd_pd(tmp[0], var7[0], var11[0]);
    var11[1] = _mm256_fmadd_pd(tmp[1], var7[1], var11[1]);
  }
  var3 = _mm256_add_pd(var11[0], var3);
  var3 = _mm256_add_pd(var11[1], var3);
  var2 = _mm256_add_pd(var10[0], var2);
  var2 = _mm256_add_pd(var10[1], var2);
  var1 = _mm256_add_pd(var9[0], var1);
  var1 = _mm256_add_pd(var9[1], var1);
  var0 = _mm256_add_pd(var8[0], var0);
  var0 = _mm256_add_pd(var8[1], var0);
  for (int_fast32_t joi = 0; joi < (((3 + n) / (4)) - 1) % 2; joi++) {
    __m256d tmp;
    __m256d var4;
    __m256d var5;
    __m256d var6;
    __m256d var7;
    tmp = _mm256_loadu_pd(&x.data[8 * ((((3 + n) / 4) - 1) / 2) + 4 * joi]);
    var4 = _mm256_loadu_pd(&A.data[(4 * io) * (A.strides[0]) + 8 * ((((3 + n) / 4) - 1) / 2) + 4 * joi]);
    var0 = _mm256_fmadd_pd(tmp, var4, var0);
    var5 = _mm256_loadu_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * ((((3 + n) / 4) - 1) / 2) + 4 * joi]);
    var1 = _mm256_fmadd_pd(tmp, var5, var1);
    var6 = _mm256_loadu_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * ((((3 + n) / 4) - 1) / 2) + 4 * joi]);
    var2 = _mm256_fmadd_pd(tmp, var6, var2);
    var7 = _mm256_loadu_pd(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 * ((((3 + n) / 4) - 1) / 2) + 4 * joi]);
    var3 = _mm256_fmadd_pd(tmp, var7, var3);
  }
  for (int_fast32_t jo = ((3 + n) / (4)) - 1; jo < ((3 + n) / (4)); jo++) {
    __m256d tmp;
    __m256d var4;
    __m256d var5;
    __m256d var6;
    __m256d var7;
    
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
            var4 = _mm256_maskload_pd(&A.data[(4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var0 = _mm256_fmadd_pd(prefixed_src1, var4, var0);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var5 = _mm256_maskload_pd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var1 = _mm256_fmadd_pd(prefixed_src1, var5, var1);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var6 = _mm256_maskload_pd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var2 = _mm256_fmadd_pd(prefixed_src1, var6, var2);
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
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), tmp, _mm256_castsi256_pd(cmp));
    var3 = _mm256_fmadd_pd(prefixed_src1, var7, var3);
}

  }
  
    {
        __m256d tmp = _mm256_hadd_pd(var3, var3);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&result_3) += _mm256_cvtsd_f64(tmp);
    }
    
  
    {
        __m256d tmp = _mm256_hadd_pd(var2, var2);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&result_2) += _mm256_cvtsd_f64(tmp);
    }
    
  
    {
        __m256d tmp = _mm256_hadd_pd(var1, var1);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&result_1) += _mm256_cvtsd_f64(tmp);
    }
    
  
    {
        __m256d tmp = _mm256_hadd_pd(var0, var0);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(&result_0) += _mm256_cvtsd_f64(tmp);
    }
    
  result_0 = alpha_ * result_0;
  y.data[4 * io] += result_0;
  result_1 = alpha_ * result_1;
  y.data[1 + 4 * io] += result_1;
  result_2 = alpha_ * result_2;
  y.data[2 + 4 * io] += result_2;
  result_3 = alpha_ * result_3;
  y.data[3 + 4 * io] += result_3;
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  y.data[ii + (m / 4) * 4] = y.data[ii + (m / 4) * 4] * beta_;
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
  result = alpha_ * result;
  y.data[ii + (m / 4) * 4] += result;
}
}

// exo_dgemv_rm_nt_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][m, n] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_rm_nt_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i * y.strides[0]] = y.data[i * y.strides[0]] * beta_;
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  result = alpha_ * result;
  y.data[i * y.strides[0]] += result;
}
}

// exo_dgemv_rm_t_stride_1(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][n, m] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_rm_t_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i] = y.data[i] * beta_;
}
for (int_fast32_t jo = 0; jo < ((n) / (4)); jo++) {
  __m256d var2;
  var2 = _mm256_broadcast_sd(&x.data[4 * jo]);
  __m256d var1;
  var1 = _mm256_broadcast_sd((&alpha_));
  __m256d var0;
  var0 = _mm256_mul_pd(var1, var2);
  __m256d var6;
  var6 = _mm256_broadcast_sd(&x.data[1 + 4 * jo]);
  __m256d var5;
  var5 = _mm256_broadcast_sd((&alpha_));
  __m256d var4;
  var4 = _mm256_mul_pd(var5, var6);
  __m256d var10;
  var10 = _mm256_broadcast_sd(&x.data[2 + 4 * jo]);
  __m256d var9;
  var9 = _mm256_broadcast_sd((&alpha_));
  __m256d var8;
  var8 = _mm256_mul_pd(var9, var10);
  __m256d var14;
  var14 = _mm256_broadcast_sd(&x.data[3 + 4 * jo]);
  __m256d var13;
  var13 = _mm256_broadcast_sd((&alpha_));
  __m256d var12;
  var12 = _mm256_mul_pd(var13, var14);
  for (int_fast32_t ioo = 0; ioo < ((((3 + m) / (4)) - 1) / (2)); ioo++) {
    __m256d tmp[2];
    __m256d var3[2];
    __m256d var7[2];
    __m256d var11[2];
    __m256d var15[2];
    tmp[0] = _mm256_loadu_pd(&y.data[8 * ioo]);
    tmp[1] = _mm256_loadu_pd(&y.data[4 + 8 * ioo]);
    var3[0] = _mm256_loadu_pd(&A.data[(4 * jo) * (A.strides[0]) + 8 * ioo]);
    var3[1] = _mm256_loadu_pd(&A.data[(4 * jo) * (A.strides[0]) + 4 + 8 * ioo]);
    tmp[0] = _mm256_fmadd_pd(var0, var3[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var0, var3[1], tmp[1]);
    var7[0] = _mm256_loadu_pd(&A.data[(1 + 4 * jo) * (A.strides[0]) + 8 * ioo]);
    var7[1] = _mm256_loadu_pd(&A.data[(1 + 4 * jo) * (A.strides[0]) + 4 + 8 * ioo]);
    tmp[0] = _mm256_fmadd_pd(var4, var7[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var4, var7[1], tmp[1]);
    var11[0] = _mm256_loadu_pd(&A.data[(2 + 4 * jo) * (A.strides[0]) + 8 * ioo]);
    var11[1] = _mm256_loadu_pd(&A.data[(2 + 4 * jo) * (A.strides[0]) + 4 + 8 * ioo]);
    tmp[0] = _mm256_fmadd_pd(var8, var11[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var8, var11[1], tmp[1]);
    var15[0] = _mm256_loadu_pd(&A.data[(3 + 4 * jo) * (A.strides[0]) + 8 * ioo]);
    var15[1] = _mm256_loadu_pd(&A.data[(3 + 4 * jo) * (A.strides[0]) + 4 + 8 * ioo]);
    tmp[0] = _mm256_fmadd_pd(var12, var15[0], tmp[0]);
    tmp[1] = _mm256_fmadd_pd(var12, var15[1], tmp[1]);
    _mm256_storeu_pd(&y.data[8 * ioo], tmp[0]);
    _mm256_storeu_pd(&y.data[4 + 8 * ioo], tmp[1]);
  }
  for (int_fast32_t ioi = 0; ioi < (((3 + m) / (4)) - 1) % 2; ioi++) {
    __m256d tmp;
    __m256d var3;
    __m256d var7;
    __m256d var11;
    __m256d var15;
    tmp = _mm256_loadu_pd(&y.data[8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi]);
    var3 = _mm256_loadu_pd(&A.data[(4 * jo) * (A.strides[0]) + 8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi]);
    tmp = _mm256_fmadd_pd(var0, var3, tmp);
    var7 = _mm256_loadu_pd(&A.data[(1 + 4 * jo) * (A.strides[0]) + 8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi]);
    tmp = _mm256_fmadd_pd(var4, var7, tmp);
    var11 = _mm256_loadu_pd(&A.data[(2 + 4 * jo) * (A.strides[0]) + 8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi]);
    tmp = _mm256_fmadd_pd(var8, var11, tmp);
    var15 = _mm256_loadu_pd(&A.data[(3 + 4 * jo) * (A.strides[0]) + 8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi]);
    tmp = _mm256_fmadd_pd(var12, var15, tmp);
    _mm256_storeu_pd(&y.data[8 * ((((3 + m) / 4) - 1) / 2) + 4 * ioi], tmp);
  }
  for (int_fast32_t io = ((3 + m) / (4)) - 1; io < ((3 + m) / (4)); io++) {
    __m256d tmp;
    __m256d var3;
    __m256d var7;
    __m256d var11;
    __m256d var15;
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            tmp = _mm256_maskload_pd(&y.data[4 * io], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var3 = _mm256_maskload_pd(&A.data[(4 * jo) * (A.strides[0]) + 4 * io], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var3, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var7 = _mm256_maskload_pd(&A.data[(1 + 4 * jo) * (A.strides[0]) + 4 * io], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var4, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var7, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var11 = _mm256_maskload_pd(&A.data[(2 + 4 * jo) * (A.strides[0]) + 4 * io], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var8, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var11, tmp);
}

    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var15 = _mm256_maskload_pd(&A.data[(3 + 4 * jo) * (A.strides[0]) + 4 * io], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var12, _mm256_castsi256_pd(cmp));
    tmp = _mm256_fmadd_pd(prefixed_src1, var15, tmp);
}

    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * io], cmp, tmp);
    }
    
  }
}
for (int_fast32_t ji = 0; ji < n % 4; ji++) {
  for (int_fast32_t i = 0; i < m; i++) {
    y.data[i] += alpha_ * x.data[ji + (n / 4) * 4] * A.data[(ji + (n / 4) * 4) * A.strides[0] + i];
  }
}
}

// exo_dgemv_rm_t_stride_any(
//     m : size,
//     n : size,
//     alpha : f64 @DRAM,
//     beta : f64 @DRAM,
//     A : [f64][n, m] @DRAM,
//     x : [f64][n] @DRAM,
//     y : [f64][m] @DRAM
// )
void exo_dgemv_rm_t_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const double* alpha, const double* beta, struct exo_win_2f64c A, struct exo_win_1f64c x, struct exo_win_1f64 y ) {
// assert stride(A, 1) == 1
double beta_;
beta_ = *beta;
double alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i * y.strides[0]] = y.data[i * y.strides[0]] * beta_;
}
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    y.data[i * y.strides[0]] += alpha_ * x.data[j * x.strides[0]] * A.data[j * A.strides[0] + i];
  }
}
}

// exo_sgemv_rm_nt_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_rm_nt_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t io = 0; io < ((m) / (4)); io++) {
  float result_0;
  float result_1;
  float result_2;
  float result_3;
  y.data[4 * io] = y.data[4 * io] * beta_;
  result_0 = 0.0;
  y.data[1 + 4 * io] = y.data[1 + 4 * io] * beta_;
  result_1 = 0.0;
  y.data[2 + 4 * io] = y.data[2 + 4 * io] * beta_;
  result_2 = 0.0;
  y.data[3 + 4 * io] = y.data[3 + 4 * io] * beta_;
  result_3 = 0.0;
  __m256 var0;
  var0 = _mm256_setzero_ps();
  __m256 var1;
  var1 = _mm256_setzero_ps();
  __m256 var2;
  var2 = _mm256_setzero_ps();
  __m256 var3;
  var3 = _mm256_setzero_ps();
  __m256 var8[2];
  var8[0] = _mm256_setzero_ps();
  var8[1] = _mm256_setzero_ps();
  __m256 var9[2];
  var9[0] = _mm256_setzero_ps();
  var9[1] = _mm256_setzero_ps();
  __m256 var10[2];
  var10[0] = _mm256_setzero_ps();
  var10[1] = _mm256_setzero_ps();
  __m256 var11[2];
  var11[0] = _mm256_setzero_ps();
  var11[1] = _mm256_setzero_ps();
  for (int_fast32_t joo = 0; joo < ((((7 + n) / (8)) - 1) / (2)); joo++) {
    __m256 tmp[2];
    __m256 var4[2];
    __m256 var5[2];
    __m256 var6[2];
    __m256 var7[2];
    tmp[0] = _mm256_loadu_ps(&x.data[16 * joo]);
    tmp[1] = _mm256_loadu_ps(&x.data[8 + 16 * joo]);
    var4[0] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 16 * joo]);
    var4[1] = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var8[0] = _mm256_fmadd_ps(tmp[0], var4[0], var8[0]);
    var8[1] = _mm256_fmadd_ps(tmp[1], var4[1], var8[1]);
    var5[0] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var5[1] = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var9[0] = _mm256_fmadd_ps(tmp[0], var5[0], var9[0]);
    var9[1] = _mm256_fmadd_ps(tmp[1], var5[1], var9[1]);
    var6[0] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var6[1] = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var10[0] = _mm256_fmadd_ps(tmp[0], var6[0], var10[0]);
    var10[1] = _mm256_fmadd_ps(tmp[1], var6[1], var10[1]);
    var7[0] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * joo]);
    var7[1] = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 8 + 16 * joo]);
    var11[0] = _mm256_fmadd_ps(tmp[0], var7[0], var11[0]);
    var11[1] = _mm256_fmadd_ps(tmp[1], var7[1], var11[1]);
  }
  var3 = _mm256_add_ps(var11[0], var3);
  var3 = _mm256_add_ps(var11[1], var3);
  var2 = _mm256_add_ps(var10[0], var2);
  var2 = _mm256_add_ps(var10[1], var2);
  var1 = _mm256_add_ps(var9[0], var1);
  var1 = _mm256_add_ps(var9[1], var1);
  var0 = _mm256_add_ps(var8[0], var0);
  var0 = _mm256_add_ps(var8[1], var0);
  for (int_fast32_t joi = 0; joi < (((7 + n) / (8)) - 1) % 2; joi++) {
    __m256 tmp;
    __m256 var4;
    __m256 var5;
    __m256 var6;
    __m256 var7;
    tmp = _mm256_loadu_ps(&x.data[16 * ((((7 + n) / 8) - 1) / 2) + 8 * joi]);
    var4 = _mm256_loadu_ps(&A.data[(4 * io) * (A.strides[0]) + 16 * ((((7 + n) / 8) - 1) / 2) + 8 * joi]);
    var0 = _mm256_fmadd_ps(tmp, var4, var0);
    var5 = _mm256_loadu_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 16 * ((((7 + n) / 8) - 1) / 2) + 8 * joi]);
    var1 = _mm256_fmadd_ps(tmp, var5, var1);
    var6 = _mm256_loadu_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 16 * ((((7 + n) / 8) - 1) / 2) + 8 * joi]);
    var2 = _mm256_fmadd_ps(tmp, var6, var2);
    var7 = _mm256_loadu_ps(&A.data[(3 + 4 * io) * (A.strides[0]) + 16 * ((((7 + n) / 8) - 1) / 2) + 8 * joi]);
    var3 = _mm256_fmadd_ps(tmp, var7, var3);
  }
  for (int_fast32_t jo = ((7 + n) / (8)) - 1; jo < ((7 + n) / (8)); jo++) {
    __m256 tmp;
    __m256 var4;
    __m256 var5;
    __m256 var6;
    __m256 var7;
    
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
    var4 = _mm256_maskload_ps(&A.data[(4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var0 = _mm256_fmadd_ps(prefixed_src1, var4, var0);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var5 = _mm256_maskload_ps(&A.data[(1 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var1 = _mm256_fmadd_ps(prefixed_src1, var5, var1);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var6 = _mm256_maskload_ps(&A.data[(2 + 4 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var2 = _mm256_fmadd_ps(prefixed_src1, var6, var2);
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
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), tmp, _mm256_castsi256_ps(cmp));
    var3 = _mm256_fmadd_ps(prefixed_src1, var7, var3);
}

  }
  
    {
        __m256 tmp = _mm256_hadd_ps(var3, var3);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(&result_3) += _mm256_cvtss_f32(tmp);
    }
    
  
    {
        __m256 tmp = _mm256_hadd_ps(var2, var2);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(&result_2) += _mm256_cvtss_f32(tmp);
    }
    
  
    {
        __m256 tmp = _mm256_hadd_ps(var1, var1);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(&result_1) += _mm256_cvtss_f32(tmp);
    }
    
  
    {
        __m256 tmp = _mm256_hadd_ps(var0, var0);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(&result_0) += _mm256_cvtss_f32(tmp);
    }
    
  result_0 = alpha_ * result_0;
  y.data[4 * io] += result_0;
  result_1 = alpha_ * result_1;
  y.data[1 + 4 * io] += result_1;
  result_2 = alpha_ * result_2;
  y.data[2 + 4 * io] += result_2;
  result_3 = alpha_ * result_3;
  y.data[3 + 4 * io] += result_3;
}
for (int_fast32_t ii = 0; ii < m % 4; ii++) {
  y.data[ii + (m / 4) * 4] = y.data[ii + (m / 4) * 4] * beta_;
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 4) * 4) * A.strides[0] + j];
  }
  result = alpha_ * result;
  y.data[ii + (m / 4) * 4] += result;
}
}

// exo_sgemv_rm_nt_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][m, n] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_rm_nt_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i * y.strides[0]] = y.data[i * y.strides[0]] * beta_;
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j * x.strides[0]] * A.data[i * A.strides[0] + j];
  }
  result = alpha_ * result;
  y.data[i * y.strides[0]] += result;
}
}

// exo_sgemv_rm_t_stride_1(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][n, m] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_rm_t_stride_1( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
// assert stride(A, 1) == 1
// assert stride(x, 0) == 1
// assert stride(y, 0) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i] = y.data[i] * beta_;
}
for (int_fast32_t jo = 0; jo < ((n) / (4)); jo++) {
  __m256 var2;
  var2 = _mm256_broadcast_ss(&x.data[4 * jo]);
  __m256 var1;
  var1 = _mm256_broadcast_ss((&alpha_));
  __m256 var0;
  var0 = _mm256_mul_ps(var1, var2);
  __m256 var6;
  var6 = _mm256_broadcast_ss(&x.data[1 + 4 * jo]);
  __m256 var5;
  var5 = _mm256_broadcast_ss((&alpha_));
  __m256 var4;
  var4 = _mm256_mul_ps(var5, var6);
  __m256 var10;
  var10 = _mm256_broadcast_ss(&x.data[2 + 4 * jo]);
  __m256 var9;
  var9 = _mm256_broadcast_ss((&alpha_));
  __m256 var8;
  var8 = _mm256_mul_ps(var9, var10);
  __m256 var14;
  var14 = _mm256_broadcast_ss(&x.data[3 + 4 * jo]);
  __m256 var13;
  var13 = _mm256_broadcast_ss((&alpha_));
  __m256 var12;
  var12 = _mm256_mul_ps(var13, var14);
  for (int_fast32_t ioo = 0; ioo < ((((7 + m) / (8)) - 1) / (2)); ioo++) {
    __m256 tmp[2];
    __m256 var3[2];
    __m256 var7[2];
    __m256 var11[2];
    __m256 var15[2];
    tmp[0] = _mm256_loadu_ps(&y.data[16 * ioo]);
    tmp[1] = _mm256_loadu_ps(&y.data[8 + 16 * ioo]);
    var3[0] = _mm256_loadu_ps(&A.data[(4 * jo) * (A.strides[0]) + 16 * ioo]);
    var3[1] = _mm256_loadu_ps(&A.data[(4 * jo) * (A.strides[0]) + 8 + 16 * ioo]);
    tmp[0] = _mm256_fmadd_ps(var0, var3[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var0, var3[1], tmp[1]);
    var7[0] = _mm256_loadu_ps(&A.data[(1 + 4 * jo) * (A.strides[0]) + 16 * ioo]);
    var7[1] = _mm256_loadu_ps(&A.data[(1 + 4 * jo) * (A.strides[0]) + 8 + 16 * ioo]);
    tmp[0] = _mm256_fmadd_ps(var4, var7[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var4, var7[1], tmp[1]);
    var11[0] = _mm256_loadu_ps(&A.data[(2 + 4 * jo) * (A.strides[0]) + 16 * ioo]);
    var11[1] = _mm256_loadu_ps(&A.data[(2 + 4 * jo) * (A.strides[0]) + 8 + 16 * ioo]);
    tmp[0] = _mm256_fmadd_ps(var8, var11[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var8, var11[1], tmp[1]);
    var15[0] = _mm256_loadu_ps(&A.data[(3 + 4 * jo) * (A.strides[0]) + 16 * ioo]);
    var15[1] = _mm256_loadu_ps(&A.data[(3 + 4 * jo) * (A.strides[0]) + 8 + 16 * ioo]);
    tmp[0] = _mm256_fmadd_ps(var12, var15[0], tmp[0]);
    tmp[1] = _mm256_fmadd_ps(var12, var15[1], tmp[1]);
    _mm256_storeu_ps(&y.data[16 * ioo], tmp[0]);
    _mm256_storeu_ps(&y.data[8 + 16 * ioo], tmp[1]);
  }
  for (int_fast32_t ioi = 0; ioi < (((7 + m) / (8)) - 1) % 2; ioi++) {
    __m256 tmp;
    __m256 var3;
    __m256 var7;
    __m256 var11;
    __m256 var15;
    tmp = _mm256_loadu_ps(&y.data[16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi]);
    var3 = _mm256_loadu_ps(&A.data[(4 * jo) * (A.strides[0]) + 16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi]);
    tmp = _mm256_fmadd_ps(var0, var3, tmp);
    var7 = _mm256_loadu_ps(&A.data[(1 + 4 * jo) * (A.strides[0]) + 16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi]);
    tmp = _mm256_fmadd_ps(var4, var7, tmp);
    var11 = _mm256_loadu_ps(&A.data[(2 + 4 * jo) * (A.strides[0]) + 16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi]);
    tmp = _mm256_fmadd_ps(var8, var11, tmp);
    var15 = _mm256_loadu_ps(&A.data[(3 + 4 * jo) * (A.strides[0]) + 16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi]);
    tmp = _mm256_fmadd_ps(var12, var15, tmp);
    _mm256_storeu_ps(&y.data[16 * ((((7 + m) / 8) - 1) / 2) + 8 * ioi], tmp);
  }
  for (int_fast32_t io = ((7 + m) / (8)) - 1; io < ((7 + m) / (8)); io++) {
    __m256 tmp;
    __m256 var3;
    __m256 var7;
    __m256 var11;
    __m256 var15;
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    tmp = _mm256_maskload_ps(&y.data[8 * io], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var3 = _mm256_maskload_ps(&A.data[(4 * jo) * (A.strides[0]) + 8 * io], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var3, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var7 = _mm256_maskload_ps(&A.data[(1 + 4 * jo) * (A.strides[0]) + 8 * io], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var4, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var7, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var11 = _mm256_maskload_ps(&A.data[(2 + 4 * jo) * (A.strides[0]) + 8 * io], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var8, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var11, tmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var15 = _mm256_maskload_ps(&A.data[(3 + 4 * jo) * (A.strides[0]) + 8 * io], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var12, _mm256_castsi256_ps(cmp));
    tmp = _mm256_fmadd_ps(prefixed_src1, var15, tmp);
}

    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * io) + m));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * io], cmp, tmp);
    }
    
  }
}
for (int_fast32_t ji = 0; ji < n % 4; ji++) {
  for (int_fast32_t i = 0; i < m; i++) {
    y.data[i] += alpha_ * x.data[ji + (n / 4) * 4] * A.data[(ji + (n / 4) * 4) * A.strides[0] + i];
  }
}
}

// exo_sgemv_rm_t_stride_any(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     A : [f32][n, m] @DRAM,
//     x : [f32][n] @DRAM,
//     y : [f32][m] @DRAM
// )
void exo_sgemv_rm_t_stride_any( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, const float* beta, struct exo_win_2f32c A, struct exo_win_1f32c x, struct exo_win_1f32 y ) {
// assert stride(A, 1) == 1
float beta_;
beta_ = *beta;
float alpha_;
alpha_ = *alpha;
for (int_fast32_t i = 0; i < m; i++) {
  y.data[i * y.strides[0]] = y.data[i * y.strides[0]] * beta_;
}
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    y.data[i * y.strides[0]] += alpha_ * x.data[j * x.strides[0]] * A.data[j * A.strides[0] + i];
  }
}
}


/* relying on the following instruction..."
mm256_broadcast_sd(out,val)
{out_data} = _mm256_broadcast_sd(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_sd_scalar(out,val)
{out_data} = _mm256_broadcast_sd({val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss_scalar(out,val)
{out_data} = _mm256_broadcast_ss({val_data});
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
mm256_mul_pd(out,x,y)
{out_data} = _mm256_mul_pd({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_mul_ps(out,x,y)
{out_data} = _mm256_mul_ps({x_data}, {y_data});
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
