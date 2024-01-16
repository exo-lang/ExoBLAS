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
for (int_fast32_t io = 0; io < ((m) / (15)); io++) {
  double *result = (double*) malloc(15 * sizeof(*result));
  __m256d var0[15];
  result[0] = 0.0;
  var0[0] = _mm256_setzero_pd();
  result[1] = 0.0;
  var0[1] = _mm256_setzero_pd();
  result[2] = 0.0;
  var0[2] = _mm256_setzero_pd();
  result[3] = 0.0;
  var0[3] = _mm256_setzero_pd();
  result[4] = 0.0;
  var0[4] = _mm256_setzero_pd();
  result[5] = 0.0;
  var0[5] = _mm256_setzero_pd();
  result[6] = 0.0;
  var0[6] = _mm256_setzero_pd();
  result[7] = 0.0;
  var0[7] = _mm256_setzero_pd();
  result[8] = 0.0;
  var0[8] = _mm256_setzero_pd();
  result[9] = 0.0;
  var0[9] = _mm256_setzero_pd();
  result[10] = 0.0;
  var0[10] = _mm256_setzero_pd();
  result[11] = 0.0;
  var0[11] = _mm256_setzero_pd();
  result[12] = 0.0;
  var0[12] = _mm256_setzero_pd();
  result[13] = 0.0;
  var0[13] = _mm256_setzero_pd();
  result[14] = 0.0;
  var0[14] = _mm256_setzero_pd();
  for (int_fast32_t jo = 0; jo < ((3 + n) / (4)) - 1; jo++) {
    __m256d shared;
    shared = _mm256_loadu_pd(&x.data[4 * jo]);
    __m256d var1[15];
    var1[0] = _mm256_loadu_pd(&A.data[(15 * io) * (A.strides[0]) + 4 * jo]);
    var1[1] = _mm256_loadu_pd(&A.data[(1 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[2] = _mm256_loadu_pd(&A.data[(2 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[3] = _mm256_loadu_pd(&A.data[(3 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[4] = _mm256_loadu_pd(&A.data[(4 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[5] = _mm256_loadu_pd(&A.data[(5 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[6] = _mm256_loadu_pd(&A.data[(6 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[7] = _mm256_loadu_pd(&A.data[(7 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[8] = _mm256_loadu_pd(&A.data[(8 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[9] = _mm256_loadu_pd(&A.data[(9 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[10] = _mm256_loadu_pd(&A.data[(10 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[11] = _mm256_loadu_pd(&A.data[(11 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[12] = _mm256_loadu_pd(&A.data[(12 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[13] = _mm256_loadu_pd(&A.data[(13 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[14] = _mm256_loadu_pd(&A.data[(14 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var0[0] = _mm256_fmadd_pd(shared, var1[0], var0[0]);
    var0[1] = _mm256_fmadd_pd(shared, var1[1], var0[1]);
    var0[2] = _mm256_fmadd_pd(shared, var1[2], var0[2]);
    var0[3] = _mm256_fmadd_pd(shared, var1[3], var0[3]);
    var0[4] = _mm256_fmadd_pd(shared, var1[4], var0[4]);
    var0[5] = _mm256_fmadd_pd(shared, var1[5], var0[5]);
    var0[6] = _mm256_fmadd_pd(shared, var1[6], var0[6]);
    var0[7] = _mm256_fmadd_pd(shared, var1[7], var0[7]);
    var0[8] = _mm256_fmadd_pd(shared, var1[8], var0[8]);
    var0[9] = _mm256_fmadd_pd(shared, var1[9], var0[9]);
    var0[10] = _mm256_fmadd_pd(shared, var1[10], var0[10]);
    var0[11] = _mm256_fmadd_pd(shared, var1[11], var0[11]);
    var0[12] = _mm256_fmadd_pd(shared, var1[12], var0[12]);
    var0[13] = _mm256_fmadd_pd(shared, var1[13], var0[13]);
    var0[14] = _mm256_fmadd_pd(shared, var1[14], var0[14]);
  }
  for (int_fast32_t jo = ((3 + n) / (4)) - 1; jo < ((3 + n) / (4)); jo++) {
    __m256d shared;
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            shared = _mm256_maskload_pd(&x.data[4 * jo], cmp);
       }
       
    __m256d var1[15];
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[0] = _mm256_maskload_pd(&A.data[(15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[1] = _mm256_maskload_pd(&A.data[(1 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[2] = _mm256_maskload_pd(&A.data[(2 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[3] = _mm256_maskload_pd(&A.data[(3 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[4] = _mm256_maskload_pd(&A.data[(4 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[5] = _mm256_maskload_pd(&A.data[(5 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[6] = _mm256_maskload_pd(&A.data[(6 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[7] = _mm256_maskload_pd(&A.data[(7 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[8] = _mm256_maskload_pd(&A.data[(8 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[9] = _mm256_maskload_pd(&A.data[(9 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[10] = _mm256_maskload_pd(&A.data[(10 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[11] = _mm256_maskload_pd(&A.data[(11 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[12] = _mm256_maskload_pd(&A.data[(12 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[13] = _mm256_maskload_pd(&A.data[(13 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[14] = _mm256_maskload_pd(&A.data[(14 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[0] = _mm256_fmadd_pd(prefixed_src1, var1[0], var0[0]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[1] = _mm256_fmadd_pd(prefixed_src1, var1[1], var0[1]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[2] = _mm256_fmadd_pd(prefixed_src1, var1[2], var0[2]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[3] = _mm256_fmadd_pd(prefixed_src1, var1[3], var0[3]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[4] = _mm256_fmadd_pd(prefixed_src1, var1[4], var0[4]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[5] = _mm256_fmadd_pd(prefixed_src1, var1[5], var0[5]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[6] = _mm256_fmadd_pd(prefixed_src1, var1[6], var0[6]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[7] = _mm256_fmadd_pd(prefixed_src1, var1[7], var0[7]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[8] = _mm256_fmadd_pd(prefixed_src1, var1[8], var0[8]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[9] = _mm256_fmadd_pd(prefixed_src1, var1[9], var0[9]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[10] = _mm256_fmadd_pd(prefixed_src1, var1[10], var0[10]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[11] = _mm256_fmadd_pd(prefixed_src1, var1[11], var0[11]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[12] = _mm256_fmadd_pd(prefixed_src1, var1[12], var0[12]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[13] = _mm256_fmadd_pd(prefixed_src1, var1[13], var0[13]);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), shared, _mm256_castsi256_pd(cmp));
    var0[14] = _mm256_fmadd_pd(prefixed_src1, var1[14], var0[14]);
}

  }
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[0], var0[0]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[0] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[15 * io] = beta_ * y.data[15 * io] + alpha_ * result[0];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[1], var0[1]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[1] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[1 + 15 * io] = beta_ * y.data[1 + 15 * io] + alpha_ * result[1];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[2], var0[2]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[2] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[2 + 15 * io] = beta_ * y.data[2 + 15 * io] + alpha_ * result[2];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[3], var0[3]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[3] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[3 + 15 * io] = beta_ * y.data[3 + 15 * io] + alpha_ * result[3];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[4], var0[4]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[4] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[4 + 15 * io] = beta_ * y.data[4 + 15 * io] + alpha_ * result[4];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[5], var0[5]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[5] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[5 + 15 * io] = beta_ * y.data[5 + 15 * io] + alpha_ * result[5];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[6], var0[6]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[6] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[6 + 15 * io] = beta_ * y.data[6 + 15 * io] + alpha_ * result[6];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[7], var0[7]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[7] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[7 + 15 * io] = beta_ * y.data[7 + 15 * io] + alpha_ * result[7];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[8], var0[8]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[8] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[8 + 15 * io] = beta_ * y.data[8 + 15 * io] + alpha_ * result[8];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[9], var0[9]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[9] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[9 + 15 * io] = beta_ * y.data[9 + 15 * io] + alpha_ * result[9];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[10], var0[10]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[10] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[10 + 15 * io] = beta_ * y.data[10 + 15 * io] + alpha_ * result[10];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[11], var0[11]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[11] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[11 + 15 * io] = beta_ * y.data[11 + 15 * io] + alpha_ * result[11];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[12], var0[12]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[12] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[12 + 15 * io] = beta_ * y.data[12 + 15 * io] + alpha_ * result[12];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[13], var0[13]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[13] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[13 + 15 * io] = beta_ * y.data[13 + 15 * io] + alpha_ * result[13];
  
    {
        __m256d tmp = _mm256_hadd_pd(var0[14], var0[14]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        result[14] += _mm256_cvtsd_f64(tmp);
    }
    
  y.data[14 + 15 * io] = beta_ * y.data[14 + 15 * io] + alpha_ * result[14];
  free(result);
}
for (int_fast32_t ii = 0; ii < m % 15; ii++) {
  double result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 15) * 15) * A.strides[0] + j];
  }
  y.data[ii + (m / 15) * 15] = beta_ * y.data[ii + (m / 15) * 15] + alpha_ * result;
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
for (int_fast32_t io = 0; io < ((m) / (15)); io++) {
  double *alphaXi = (double*) malloc(15 * sizeof(*alphaXi));
  alphaXi[0] = alpha_ * x.data[15 * io];
  alphaXi[1] = alpha_ * x.data[1 + 15 * io];
  alphaXi[2] = alpha_ * x.data[2 + 15 * io];
  alphaXi[3] = alpha_ * x.data[3 + 15 * io];
  alphaXi[4] = alpha_ * x.data[4 + 15 * io];
  alphaXi[5] = alpha_ * x.data[5 + 15 * io];
  alphaXi[6] = alpha_ * x.data[6 + 15 * io];
  alphaXi[7] = alpha_ * x.data[7 + 15 * io];
  alphaXi[8] = alpha_ * x.data[8 + 15 * io];
  alphaXi[9] = alpha_ * x.data[9 + 15 * io];
  alphaXi[10] = alpha_ * x.data[10 + 15 * io];
  alphaXi[11] = alpha_ * x.data[11 + 15 * io];
  alphaXi[12] = alpha_ * x.data[12 + 15 * io];
  alphaXi[13] = alpha_ * x.data[13 + 15 * io];
  alphaXi[14] = alpha_ * x.data[14 + 15 * io];
  for (int_fast32_t jo = 0; jo < ((3 + n) / (4)) - 1; jo++) {
    __m256d shared;
    shared = _mm256_loadu_pd(&y.data[4 * jo]);
    __m256d var0[15];
    __m256d var1[15];
    var0[0] = _mm256_broadcast_sd(&alphaXi[0]);
    var0[1] = _mm256_broadcast_sd(&alphaXi[1]);
    var0[2] = _mm256_broadcast_sd(&alphaXi[2]);
    var0[3] = _mm256_broadcast_sd(&alphaXi[3]);
    var0[4] = _mm256_broadcast_sd(&alphaXi[4]);
    var0[5] = _mm256_broadcast_sd(&alphaXi[5]);
    var0[6] = _mm256_broadcast_sd(&alphaXi[6]);
    var0[7] = _mm256_broadcast_sd(&alphaXi[7]);
    var0[8] = _mm256_broadcast_sd(&alphaXi[8]);
    var0[9] = _mm256_broadcast_sd(&alphaXi[9]);
    var0[10] = _mm256_broadcast_sd(&alphaXi[10]);
    var0[11] = _mm256_broadcast_sd(&alphaXi[11]);
    var0[12] = _mm256_broadcast_sd(&alphaXi[12]);
    var0[13] = _mm256_broadcast_sd(&alphaXi[13]);
    var0[14] = _mm256_broadcast_sd(&alphaXi[14]);
    var1[0] = _mm256_loadu_pd(&A.data[(15 * io) * (A.strides[0]) + 4 * jo]);
    var1[1] = _mm256_loadu_pd(&A.data[(1 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[2] = _mm256_loadu_pd(&A.data[(2 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[3] = _mm256_loadu_pd(&A.data[(3 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[4] = _mm256_loadu_pd(&A.data[(4 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[5] = _mm256_loadu_pd(&A.data[(5 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[6] = _mm256_loadu_pd(&A.data[(6 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[7] = _mm256_loadu_pd(&A.data[(7 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[8] = _mm256_loadu_pd(&A.data[(8 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[9] = _mm256_loadu_pd(&A.data[(9 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[10] = _mm256_loadu_pd(&A.data[(10 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[11] = _mm256_loadu_pd(&A.data[(11 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[12] = _mm256_loadu_pd(&A.data[(12 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[13] = _mm256_loadu_pd(&A.data[(13 + 15 * io) * (A.strides[0]) + 4 * jo]);
    var1[14] = _mm256_loadu_pd(&A.data[(14 + 15 * io) * (A.strides[0]) + 4 * jo]);
    shared = _mm256_fmadd_pd(var0[0], var1[0], shared);
    shared = _mm256_fmadd_pd(var0[1], var1[1], shared);
    shared = _mm256_fmadd_pd(var0[2], var1[2], shared);
    shared = _mm256_fmadd_pd(var0[3], var1[3], shared);
    shared = _mm256_fmadd_pd(var0[4], var1[4], shared);
    shared = _mm256_fmadd_pd(var0[5], var1[5], shared);
    shared = _mm256_fmadd_pd(var0[6], var1[6], shared);
    shared = _mm256_fmadd_pd(var0[7], var1[7], shared);
    shared = _mm256_fmadd_pd(var0[8], var1[8], shared);
    shared = _mm256_fmadd_pd(var0[9], var1[9], shared);
    shared = _mm256_fmadd_pd(var0[10], var1[10], shared);
    shared = _mm256_fmadd_pd(var0[11], var1[11], shared);
    shared = _mm256_fmadd_pd(var0[12], var1[12], shared);
    shared = _mm256_fmadd_pd(var0[13], var1[13], shared);
    shared = _mm256_fmadd_pd(var0[14], var1[14], shared);
    _mm256_storeu_pd(&y.data[4 * jo], shared);
  }
  for (int_fast32_t jo = ((3 + n) / (4)) - 1; jo < ((3 + n) / (4)); jo++) {
    __m256d shared;
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            shared = _mm256_maskload_pd(&y.data[4 * jo], cmp);
       }
       
    __m256d var0[15];
    __m256d var1[15];
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[0] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[0]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[1] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[1]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[2] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[2]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[3] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[3]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[4] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[4]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[5] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[5]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[6] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[6]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[7] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[7]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[8] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[8]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[9] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[9]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[10] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[10]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[11] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[11]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[12] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[12]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[13] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[13]), _mm256_castsi256_pd(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    var0[14] = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&alphaXi[14]), _mm256_castsi256_pd(cmp));
    }
    
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[0] = _mm256_maskload_pd(&A.data[(15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[1] = _mm256_maskload_pd(&A.data[(1 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[2] = _mm256_maskload_pd(&A.data[(2 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[3] = _mm256_maskload_pd(&A.data[(3 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[4] = _mm256_maskload_pd(&A.data[(4 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[5] = _mm256_maskload_pd(&A.data[(5 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[6] = _mm256_maskload_pd(&A.data[(6 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[7] = _mm256_maskload_pd(&A.data[(7 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[8] = _mm256_maskload_pd(&A.data[(8 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[9] = _mm256_maskload_pd(&A.data[(9 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[10] = _mm256_maskload_pd(&A.data[(10 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[11] = _mm256_maskload_pd(&A.data[(11 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[12] = _mm256_maskload_pd(&A.data[(12 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[13] = _mm256_maskload_pd(&A.data[(13 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
       {
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            var1[14] = _mm256_maskload_pd(&A.data[(14 + 15 * io) * (A.strides[0]) + 4 * jo], cmp);
       }
       
    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[0], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[0], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[1], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[1], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[2], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[2], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[3], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[3], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[4], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[4], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[5], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[5], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[6], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[6], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[7], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[7], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[8], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[8], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[9], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[9], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[10], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[10], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[11], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[11], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[12], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[12], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[13], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[13], shared);
}

    
{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), var0[14], _mm256_castsi256_pd(cmp));
    shared = _mm256_fmadd_pd(prefixed_src1, var1[14], shared);
}

    
    {
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x((-(4 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&y.data[4 * jo], cmp, shared);
    }
    
  }
  free(alphaXi);
}
for (int_fast32_t ii = 0; ii < m % 15; ii++) {
  double alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 15) * 15];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[(ii + (m / 15) * 15) * A.strides[0] + j];
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
for (int_fast32_t io = 0; io < ((m) / (15)); io++) {
  float *result = (float*) malloc(15 * sizeof(*result));
  __m256 var0[15];
  result[0] = 0.0;
  var0[0] = _mm256_setzero_ps();
  result[1] = 0.0;
  var0[1] = _mm256_setzero_ps();
  result[2] = 0.0;
  var0[2] = _mm256_setzero_ps();
  result[3] = 0.0;
  var0[3] = _mm256_setzero_ps();
  result[4] = 0.0;
  var0[4] = _mm256_setzero_ps();
  result[5] = 0.0;
  var0[5] = _mm256_setzero_ps();
  result[6] = 0.0;
  var0[6] = _mm256_setzero_ps();
  result[7] = 0.0;
  var0[7] = _mm256_setzero_ps();
  result[8] = 0.0;
  var0[8] = _mm256_setzero_ps();
  result[9] = 0.0;
  var0[9] = _mm256_setzero_ps();
  result[10] = 0.0;
  var0[10] = _mm256_setzero_ps();
  result[11] = 0.0;
  var0[11] = _mm256_setzero_ps();
  result[12] = 0.0;
  var0[12] = _mm256_setzero_ps();
  result[13] = 0.0;
  var0[13] = _mm256_setzero_ps();
  result[14] = 0.0;
  var0[14] = _mm256_setzero_ps();
  for (int_fast32_t jo = 0; jo < ((7 + n) / (8)) - 1; jo++) {
    __m256 shared;
    shared = _mm256_loadu_ps(&x.data[8 * jo]);
    __m256 var1[15];
    var1[0] = _mm256_loadu_ps(&A.data[(15 * io) * (A.strides[0]) + 8 * jo]);
    var1[1] = _mm256_loadu_ps(&A.data[(1 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[2] = _mm256_loadu_ps(&A.data[(2 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[3] = _mm256_loadu_ps(&A.data[(3 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[4] = _mm256_loadu_ps(&A.data[(4 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[5] = _mm256_loadu_ps(&A.data[(5 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[6] = _mm256_loadu_ps(&A.data[(6 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[7] = _mm256_loadu_ps(&A.data[(7 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[8] = _mm256_loadu_ps(&A.data[(8 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[9] = _mm256_loadu_ps(&A.data[(9 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[10] = _mm256_loadu_ps(&A.data[(10 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[11] = _mm256_loadu_ps(&A.data[(11 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[12] = _mm256_loadu_ps(&A.data[(12 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[13] = _mm256_loadu_ps(&A.data[(13 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[14] = _mm256_loadu_ps(&A.data[(14 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var0[0] = _mm256_fmadd_ps(shared, var1[0], var0[0]);
    var0[1] = _mm256_fmadd_ps(shared, var1[1], var0[1]);
    var0[2] = _mm256_fmadd_ps(shared, var1[2], var0[2]);
    var0[3] = _mm256_fmadd_ps(shared, var1[3], var0[3]);
    var0[4] = _mm256_fmadd_ps(shared, var1[4], var0[4]);
    var0[5] = _mm256_fmadd_ps(shared, var1[5], var0[5]);
    var0[6] = _mm256_fmadd_ps(shared, var1[6], var0[6]);
    var0[7] = _mm256_fmadd_ps(shared, var1[7], var0[7]);
    var0[8] = _mm256_fmadd_ps(shared, var1[8], var0[8]);
    var0[9] = _mm256_fmadd_ps(shared, var1[9], var0[9]);
    var0[10] = _mm256_fmadd_ps(shared, var1[10], var0[10]);
    var0[11] = _mm256_fmadd_ps(shared, var1[11], var0[11]);
    var0[12] = _mm256_fmadd_ps(shared, var1[12], var0[12]);
    var0[13] = _mm256_fmadd_ps(shared, var1[13], var0[13]);
    var0[14] = _mm256_fmadd_ps(shared, var1[14], var0[14]);
  }
  for (int_fast32_t jo = ((7 + n) / (8)) - 1; jo < ((7 + n) / (8)); jo++) {
    __m256 shared;
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    shared = _mm256_maskload_ps(&x.data[8 * jo], cmp);
}

    __m256 var1[15];
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[0] = _mm256_maskload_ps(&A.data[(15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[1] = _mm256_maskload_ps(&A.data[(1 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[2] = _mm256_maskload_ps(&A.data[(2 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[3] = _mm256_maskload_ps(&A.data[(3 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[4] = _mm256_maskload_ps(&A.data[(4 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[5] = _mm256_maskload_ps(&A.data[(5 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[6] = _mm256_maskload_ps(&A.data[(6 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[7] = _mm256_maskload_ps(&A.data[(7 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[8] = _mm256_maskload_ps(&A.data[(8 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[9] = _mm256_maskload_ps(&A.data[(9 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[10] = _mm256_maskload_ps(&A.data[(10 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[11] = _mm256_maskload_ps(&A.data[(11 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[12] = _mm256_maskload_ps(&A.data[(12 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[13] = _mm256_maskload_ps(&A.data[(13 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[14] = _mm256_maskload_ps(&A.data[(14 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[0] = _mm256_fmadd_ps(prefixed_src1, var1[0], var0[0]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[1] = _mm256_fmadd_ps(prefixed_src1, var1[1], var0[1]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[2] = _mm256_fmadd_ps(prefixed_src1, var1[2], var0[2]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[3] = _mm256_fmadd_ps(prefixed_src1, var1[3], var0[3]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[4] = _mm256_fmadd_ps(prefixed_src1, var1[4], var0[4]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[5] = _mm256_fmadd_ps(prefixed_src1, var1[5], var0[5]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[6] = _mm256_fmadd_ps(prefixed_src1, var1[6], var0[6]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[7] = _mm256_fmadd_ps(prefixed_src1, var1[7], var0[7]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[8] = _mm256_fmadd_ps(prefixed_src1, var1[8], var0[8]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[9] = _mm256_fmadd_ps(prefixed_src1, var1[9], var0[9]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[10] = _mm256_fmadd_ps(prefixed_src1, var1[10], var0[10]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[11] = _mm256_fmadd_ps(prefixed_src1, var1[11], var0[11]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[12] = _mm256_fmadd_ps(prefixed_src1, var1[12], var0[12]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[13] = _mm256_fmadd_ps(prefixed_src1, var1[13], var0[13]);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), shared, _mm256_castsi256_ps(cmp));
    var0[14] = _mm256_fmadd_ps(prefixed_src1, var1[14], var0[14]);
}

  }
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[0], var0[0]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[0] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[15 * io] = beta_ * y.data[15 * io] + alpha_ * result[0];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[1], var0[1]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[1] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[1 + 15 * io] = beta_ * y.data[1 + 15 * io] + alpha_ * result[1];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[2], var0[2]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[2] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[2 + 15 * io] = beta_ * y.data[2 + 15 * io] + alpha_ * result[2];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[3], var0[3]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[3] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[3 + 15 * io] = beta_ * y.data[3 + 15 * io] + alpha_ * result[3];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[4], var0[4]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[4] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[4 + 15 * io] = beta_ * y.data[4 + 15 * io] + alpha_ * result[4];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[5], var0[5]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[5] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[5 + 15 * io] = beta_ * y.data[5 + 15 * io] + alpha_ * result[5];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[6], var0[6]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[6] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[6 + 15 * io] = beta_ * y.data[6 + 15 * io] + alpha_ * result[6];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[7], var0[7]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[7] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[7 + 15 * io] = beta_ * y.data[7 + 15 * io] + alpha_ * result[7];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[8], var0[8]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[8] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[8 + 15 * io] = beta_ * y.data[8 + 15 * io] + alpha_ * result[8];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[9], var0[9]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[9] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[9 + 15 * io] = beta_ * y.data[9 + 15 * io] + alpha_ * result[9];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[10], var0[10]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[10] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[10 + 15 * io] = beta_ * y.data[10 + 15 * io] + alpha_ * result[10];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[11], var0[11]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[11] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[11 + 15 * io] = beta_ * y.data[11 + 15 * io] + alpha_ * result[11];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[12], var0[12]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[12] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[12 + 15 * io] = beta_ * y.data[12 + 15 * io] + alpha_ * result[12];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[13], var0[13]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[13] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[13 + 15 * io] = beta_ * y.data[13 + 15 * io] + alpha_ * result[13];
  
    {
        __m256 tmp = _mm256_hadd_ps(var0[14], var0[14]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        result[14] += _mm256_cvtss_f32(tmp);
    }
    
  y.data[14 + 15 * io] = beta_ * y.data[14 + 15 * io] + alpha_ * result[14];
  free(result);
}
for (int_fast32_t ii = 0; ii < m % 15; ii++) {
  float result;
  result = 0.0;
  for (int_fast32_t j = 0; j < n; j++) {
    result += x.data[j] * A.data[(ii + (m / 15) * 15) * A.strides[0] + j];
  }
  y.data[ii + (m / 15) * 15] = beta_ * y.data[ii + (m / 15) * 15] + alpha_ * result;
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
for (int_fast32_t io = 0; io < ((m) / (15)); io++) {
  float *alphaXi = (float*) malloc(15 * sizeof(*alphaXi));
  alphaXi[0] = alpha_ * x.data[15 * io];
  alphaXi[1] = alpha_ * x.data[1 + 15 * io];
  alphaXi[2] = alpha_ * x.data[2 + 15 * io];
  alphaXi[3] = alpha_ * x.data[3 + 15 * io];
  alphaXi[4] = alpha_ * x.data[4 + 15 * io];
  alphaXi[5] = alpha_ * x.data[5 + 15 * io];
  alphaXi[6] = alpha_ * x.data[6 + 15 * io];
  alphaXi[7] = alpha_ * x.data[7 + 15 * io];
  alphaXi[8] = alpha_ * x.data[8 + 15 * io];
  alphaXi[9] = alpha_ * x.data[9 + 15 * io];
  alphaXi[10] = alpha_ * x.data[10 + 15 * io];
  alphaXi[11] = alpha_ * x.data[11 + 15 * io];
  alphaXi[12] = alpha_ * x.data[12 + 15 * io];
  alphaXi[13] = alpha_ * x.data[13 + 15 * io];
  alphaXi[14] = alpha_ * x.data[14 + 15 * io];
  for (int_fast32_t jo = 0; jo < ((7 + n) / (8)) - 1; jo++) {
    __m256 shared;
    shared = _mm256_loadu_ps(&y.data[8 * jo]);
    __m256 var0[15];
    __m256 var1[15];
    var0[0] = _mm256_broadcast_ss(&alphaXi[0]);
    var0[1] = _mm256_broadcast_ss(&alphaXi[1]);
    var0[2] = _mm256_broadcast_ss(&alphaXi[2]);
    var0[3] = _mm256_broadcast_ss(&alphaXi[3]);
    var0[4] = _mm256_broadcast_ss(&alphaXi[4]);
    var0[5] = _mm256_broadcast_ss(&alphaXi[5]);
    var0[6] = _mm256_broadcast_ss(&alphaXi[6]);
    var0[7] = _mm256_broadcast_ss(&alphaXi[7]);
    var0[8] = _mm256_broadcast_ss(&alphaXi[8]);
    var0[9] = _mm256_broadcast_ss(&alphaXi[9]);
    var0[10] = _mm256_broadcast_ss(&alphaXi[10]);
    var0[11] = _mm256_broadcast_ss(&alphaXi[11]);
    var0[12] = _mm256_broadcast_ss(&alphaXi[12]);
    var0[13] = _mm256_broadcast_ss(&alphaXi[13]);
    var0[14] = _mm256_broadcast_ss(&alphaXi[14]);
    var1[0] = _mm256_loadu_ps(&A.data[(15 * io) * (A.strides[0]) + 8 * jo]);
    var1[1] = _mm256_loadu_ps(&A.data[(1 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[2] = _mm256_loadu_ps(&A.data[(2 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[3] = _mm256_loadu_ps(&A.data[(3 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[4] = _mm256_loadu_ps(&A.data[(4 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[5] = _mm256_loadu_ps(&A.data[(5 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[6] = _mm256_loadu_ps(&A.data[(6 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[7] = _mm256_loadu_ps(&A.data[(7 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[8] = _mm256_loadu_ps(&A.data[(8 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[9] = _mm256_loadu_ps(&A.data[(9 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[10] = _mm256_loadu_ps(&A.data[(10 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[11] = _mm256_loadu_ps(&A.data[(11 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[12] = _mm256_loadu_ps(&A.data[(12 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[13] = _mm256_loadu_ps(&A.data[(13 + 15 * io) * (A.strides[0]) + 8 * jo]);
    var1[14] = _mm256_loadu_ps(&A.data[(14 + 15 * io) * (A.strides[0]) + 8 * jo]);
    shared = _mm256_fmadd_ps(var0[0], var1[0], shared);
    shared = _mm256_fmadd_ps(var0[1], var1[1], shared);
    shared = _mm256_fmadd_ps(var0[2], var1[2], shared);
    shared = _mm256_fmadd_ps(var0[3], var1[3], shared);
    shared = _mm256_fmadd_ps(var0[4], var1[4], shared);
    shared = _mm256_fmadd_ps(var0[5], var1[5], shared);
    shared = _mm256_fmadd_ps(var0[6], var1[6], shared);
    shared = _mm256_fmadd_ps(var0[7], var1[7], shared);
    shared = _mm256_fmadd_ps(var0[8], var1[8], shared);
    shared = _mm256_fmadd_ps(var0[9], var1[9], shared);
    shared = _mm256_fmadd_ps(var0[10], var1[10], shared);
    shared = _mm256_fmadd_ps(var0[11], var1[11], shared);
    shared = _mm256_fmadd_ps(var0[12], var1[12], shared);
    shared = _mm256_fmadd_ps(var0[13], var1[13], shared);
    shared = _mm256_fmadd_ps(var0[14], var1[14], shared);
    _mm256_storeu_ps(&y.data[8 * jo], shared);
  }
  for (int_fast32_t jo = ((7 + n) / (8)) - 1; jo < ((7 + n) / (8)); jo++) {
    __m256 shared;
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    shared = _mm256_maskload_ps(&y.data[8 * jo], cmp);
}

    __m256 var0[15];
    __m256 var1[15];
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[0] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[0]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[1] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[1]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[2] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[2]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[3] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[3]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[4] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[4]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[5] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[5]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[6] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[6]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[7] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[7]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[8] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[8]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[9] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[9]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[10] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[10]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[11] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[11]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[12] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[12]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[13] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[13]), _mm256_castsi256_ps(cmp));
    }
    
    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var0[14] = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&alphaXi[14]), _mm256_castsi256_ps(cmp));
    }
    
    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[0] = _mm256_maskload_ps(&A.data[(15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[1] = _mm256_maskload_ps(&A.data[(1 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[2] = _mm256_maskload_ps(&A.data[(2 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[3] = _mm256_maskload_ps(&A.data[(3 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[4] = _mm256_maskload_ps(&A.data[(4 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[5] = _mm256_maskload_ps(&A.data[(5 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[6] = _mm256_maskload_ps(&A.data[(6 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[7] = _mm256_maskload_ps(&A.data[(7 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[8] = _mm256_maskload_ps(&A.data[(8 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[9] = _mm256_maskload_ps(&A.data[(9 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[10] = _mm256_maskload_ps(&A.data[(10 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[11] = _mm256_maskload_ps(&A.data[(11 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[12] = _mm256_maskload_ps(&A.data[(12 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[13] = _mm256_maskload_ps(&A.data[(13 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var1[14] = _mm256_maskload_ps(&A.data[(14 + 15 * io) * (A.strides[0]) + 8 * jo], cmp);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[0], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[0], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[1], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[1], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[2], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[2], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[3], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[3], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[4], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[4], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[5], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[5], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[6], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[6], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[7], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[7], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[8], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[8], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[9], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[9], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[10], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[10], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[11], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[11], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[12], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[12], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[13], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[13], shared);
}

    
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), var0[14], _mm256_castsi256_ps(cmp));
    shared = _mm256_fmadd_ps(prefixed_src1, var1[14], shared);
}

    
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * jo) + n));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&y.data[8 * jo], cmp, shared);
    }
    
  }
  free(alphaXi);
}
for (int_fast32_t ii = 0; ii < m % 15; ii++) {
  float alphaXi;
  alphaXi = alpha_ * x.data[ii + (m / 15) * 15];
  for (int_fast32_t j = 0; j < n; j++) {
    y.data[j] += alphaXi * A.data[(ii + (m / 15) * 15) * A.strides[0] + j];
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
