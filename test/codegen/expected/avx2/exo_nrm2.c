#include "exo_nrm2.h"



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
avx2_reg_copy_pd(dst,src)
{dst_data} = {src_data};
*/

/* relying on the following instruction..."
avx2_reg_copy_ps(dst,src)
{dst_data} = {src_data};
*/
// exo_dnrm2_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dnrm2_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result ) {
// assert stride(x, 0) == 1
*result = 0.0;
__m256d resultReg[4];
resultReg[0] = _mm256_setzero_pd();
resultReg[1] = _mm256_setzero_pd();
resultReg[2] = _mm256_setzero_pd();
resultReg[3] = _mm256_setzero_pd();
for (int_fast32_t io = 0; io < ((n) / (16)); io++) {
  __m256d xReg[4];
  __m256d xReg1[4];
  xReg[0] = _mm256_loadu_pd(&x.data[16 * io]);
  xReg[1] = _mm256_loadu_pd(&x.data[4 + 16 * io]);
  xReg[2] = _mm256_loadu_pd(&x.data[8 + 16 * io]);
  xReg[3] = _mm256_loadu_pd(&x.data[12 + 16 * io]);
  xReg1[0] = xReg[0];
  xReg1[1] = xReg[1];
  xReg1[2] = xReg[2];
  xReg1[3] = xReg[3];
  resultReg[0] = _mm256_fmadd_pd(xReg[0], xReg1[0], resultReg[0]);
  resultReg[1] = _mm256_fmadd_pd(xReg[1], xReg1[1], resultReg[1]);
  resultReg[2] = _mm256_fmadd_pd(xReg[2], xReg1[2], resultReg[2]);
  resultReg[3] = _mm256_fmadd_pd(xReg[3], xReg1[3], resultReg[3]);
}

    {
        __m256d tmp = _mm256_hadd_pd(resultReg[0], resultReg[0]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(result) += _mm256_cvtsd_f64(tmp);
    }
    

    {
        __m256d tmp = _mm256_hadd_pd(resultReg[1], resultReg[1]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(result) += _mm256_cvtsd_f64(tmp);
    }
    

    {
        __m256d tmp = _mm256_hadd_pd(resultReg[2], resultReg[2]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(result) += _mm256_cvtsd_f64(tmp);
    }
    

    {
        __m256d tmp = _mm256_hadd_pd(resultReg[3], resultReg[3]);
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *(result) += _mm256_cvtsd_f64(tmp);
    }
    
for (int_fast32_t ii = 0; ii < n % 16; ii++) {
  *result += x.data[ii + (n / 16) * 16] * x.data[ii + (n / 16) * 16];
}
}

// exo_dnrm2_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     result : f64 @DRAM
// )
void exo_dnrm2_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, double* result ) {
*result = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  *result += x.data[i * x.strides[0]] * x.data[i * x.strides[0]];
}
}

// exo_snrm2_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_snrm2_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result ) {
// assert stride(x, 0) == 1
*result = 0.0;
__m256 resultReg[4];
resultReg[0] = _mm256_setzero_ps();
resultReg[1] = _mm256_setzero_ps();
resultReg[2] = _mm256_setzero_ps();
resultReg[3] = _mm256_setzero_ps();
for (int_fast32_t io = 0; io < ((n) / (32)); io++) {
  __m256 xReg[4];
  __m256 xReg1[4];
  xReg[0] = _mm256_loadu_ps(&x.data[32 * io]);
  xReg[1] = _mm256_loadu_ps(&x.data[8 + 32 * io]);
  xReg[2] = _mm256_loadu_ps(&x.data[16 + 32 * io]);
  xReg[3] = _mm256_loadu_ps(&x.data[24 + 32 * io]);
  xReg1[0] = xReg[0];
  xReg1[1] = xReg[1];
  xReg1[2] = xReg[2];
  xReg1[3] = xReg[3];
  resultReg[0] = _mm256_fmadd_ps(xReg[0], xReg1[0], resultReg[0]);
  resultReg[1] = _mm256_fmadd_ps(xReg[1], xReg1[1], resultReg[1]);
  resultReg[2] = _mm256_fmadd_ps(xReg[2], xReg1[2], resultReg[2]);
  resultReg[3] = _mm256_fmadd_ps(xReg[3], xReg1[3], resultReg[3]);
}

    {
        __m256 tmp = _mm256_hadd_ps(resultReg[0], resultReg[0]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(result) += _mm256_cvtss_f32(tmp);
    }
    

    {
        __m256 tmp = _mm256_hadd_ps(resultReg[1], resultReg[1]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(result) += _mm256_cvtss_f32(tmp);
    }
    

    {
        __m256 tmp = _mm256_hadd_ps(resultReg[2], resultReg[2]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(result) += _mm256_cvtss_f32(tmp);
    }
    

    {
        __m256 tmp = _mm256_hadd_ps(resultReg[3], resultReg[3]);
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *(result) += _mm256_cvtss_f32(tmp);
    }
    
for (int_fast32_t ii = 0; ii < n % 32; ii++) {
  *result += x.data[ii + (n / 32) * 32] * x.data[ii + (n / 32) * 32];
}
}

// exo_snrm2_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     result : f32 @DRAM
// )
void exo_snrm2_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, float* result ) {
*result = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  *result += x.data[i * x.strides[0]] * x.data[i * x.strides[0]];
}
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
mm256_setzero_pd(dst)
{dst_data} = _mm256_setzero_pd();
*/

/* relying on the following instruction..."
mm256_setzero_ps(dst)
{dst_data} = _mm256_setzero_ps();
*/
