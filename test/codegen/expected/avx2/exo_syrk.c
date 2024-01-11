#include "exo_syrk.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>


// avx2_microkernel_4x16_1(
//     C : [f32][4, 16] @DRAM,
//     A : [f32][4, 256] @DRAM,
//     B : [f32][256, 16] @DRAM
// )
static void avx2_microkernel_4x16_1( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// avx2_microkernel_4x8_3(
//     C : [f64][4, 8] @DRAM,
//     A : [f64][4, 256] @DRAM,
//     B : [f64][256, 8] @DRAM
// )
static void avx2_microkernel_4x8_3( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B );

// d_apply_beta_128_256(
//     M : size,
//     N : size,
//     scalar : f64[1] @DRAM,
//     P : f64[M, M] @DRAM
// )
static void d_apply_beta_128_256( void *ctxt, int_fast32_t M, int_fast32_t N, const double* scalar, double* P );

// d_diag_handler_scheduled(
//     A1 : [f64][128, 256] @DRAM,
//     A2 : [f64][256, 128] @DRAM,
//     C : [f64][128, 128] @DRAM
// )
static void d_diag_handler_scheduled( void *ctxt, struct exo_win_2f64c A1, struct exo_win_2f64c A2, struct exo_win_2f64 C );

// dexo_dsyrk_lower_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C );

// dexo_dsyrk_lower_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C );

// dexo_dsyrk_lower_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C );

// dexo_dsyrk_lower_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C );

// dexo_dsyrk_upper_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C );

// dexo_dsyrk_upper_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C );

// dexo_dsyrk_upper_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C );

// dexo_dsyrk_upper_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C );

// dsyrk_apply_scalar_upper(
//     M : size,
//     N : size,
//     scalar : f64[1] @DRAM,
//     P : f64[M, M] @DRAM
// )
static void dsyrk_apply_scalar_upper( void *ctxt, int_fast32_t M, int_fast32_t N, const double* scalar, double* P );

// gebp_128x256_0(
//     C : [f32][128, 128] @DRAM,
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM
// )
static void gebp_128x256_0( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// gebp_128x256_2(
//     C : [f32][128, 128] @DRAM,
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM
// )
static void gebp_128x256_2( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// gebp_128x256_3(
//     C : [f64][128, 128] @DRAM,
//     A : [f64][128, 256] @DRAM,
//     B : [f64][256, 128] @DRAM
// )
static void gebp_128x256_3( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B );

// gebp_32x256_1(
//     C : [f32][32, 32] @DRAM,
//     A : [f32][32, 256] @DRAM,
//     B : [f32][256, 32] @DRAM
// )
static void gebp_32x256_1( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// gebp_32x256_4(
//     C : [f64][32, 32] @DRAM,
//     A : [f64][32, 256] @DRAM,
//     B : [f64][256, 32] @DRAM
// )
static void gebp_32x256_4( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B );

// gepp_dsyrk_scheduled(
//     N : size,
//     A1 : [f64][N, 256] @DRAM,
//     A2 : [f64][256, N] @DRAM,
//     C : [f64][N, N] @DRAM
// )
static void gepp_dsyrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f64c A1, struct exo_win_2f64c A2, struct exo_win_2f64 C );

// gepp_ssyrk_scheduled(
//     N : size,
//     A1 : [f32][N, 256] @DRAM,
//     A2 : [f32][256, N] @DRAM,
//     C : [f32][N, N] @DRAM
// )
static void gepp_ssyrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C );

// s_apply_beta_128_256(
//     M : size,
//     N : size,
//     scalar : f32[1] @DRAM,
//     P : f32[M, M] @DRAM
// )
static void s_apply_beta_128_256( void *ctxt, int_fast32_t M, int_fast32_t N, const float* scalar, float* P );

// s_diag_handler_scheduled(
//     A1 : [f32][128, 256] @DRAM,
//     A2 : [f32][256, 128] @DRAM,
//     C : [f32][128, 128] @DRAM
// )
static void s_diag_handler_scheduled( void *ctxt, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C );

// s_unsafe_microkernel_scheduled(
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM,
//     C : [f32][128, 128] @DRAM
// )
static void s_unsafe_microkernel_scheduled( void *ctxt, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C );

// sexo_ssyrk_lower_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C );

// sexo_ssyrk_lower_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C );

// sexo_ssyrk_lower_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C );

// sexo_ssyrk_lower_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C );

// sexo_ssyrk_upper_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C );

// sexo_ssyrk_upper_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C );

// sexo_ssyrk_upper_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C );

// sexo_ssyrk_upper_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C );

// ssyrk_apply_scalar_upper(
//     M : size,
//     N : size,
//     scalar : f32[1] @DRAM,
//     P : f32[M, M] @DRAM
// )
static void ssyrk_apply_scalar_upper( void *ctxt, int_fast32_t M, int_fast32_t N, const float* scalar, float* P );

// avx2_microkernel_4x16_1(
//     C : [f32][4, 16] @DRAM,
//     A : [f32][4, 256] @DRAM,
//     B : [f32][256, 16] @DRAM
// )
static void avx2_microkernel_4x16_1( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
// assert stride(C, 1) == 1
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
__m256 C_reg[4][2];
C_reg[0][0] = _mm256_loadu_ps(&C.data[0]);
C_reg[0][1] = _mm256_loadu_ps(&C.data[8]);
C_reg[1][0] = _mm256_loadu_ps(&C.data[C.strides[0]]);
C_reg[1][1] = _mm256_loadu_ps(&C.data[C.strides[0] + 8]);
C_reg[2][0] = _mm256_loadu_ps(&C.data[(2) * (C.strides[0])]);
C_reg[2][1] = _mm256_loadu_ps(&C.data[(2) * (C.strides[0]) + 8]);
C_reg[3][0] = _mm256_loadu_ps(&C.data[(3) * (C.strides[0])]);
C_reg[3][1] = _mm256_loadu_ps(&C.data[(3) * (C.strides[0]) + 8]);
for (int_fast32_t ko = 0; ko < 64; ko++) {
  __m256 var0[2];
  __m256 var1[2];
  var1[0] = _mm256_loadu_ps(&B.data[(4 * ko) * (B.strides[0])]);
  var1[1] = _mm256_loadu_ps(&B.data[(4 * ko) * (B.strides[0]) + 8]);
  var0[0] = _mm256_broadcast_ss(&A.data[4 * ko]);
  var0[1] = _mm256_broadcast_ss(&A.data[4 * ko]);
  C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
  var0[0] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko]);
  var0[1] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko]);
  C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
  var0[0] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko]);
  var0[1] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko]);
  C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
  var0[0] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko]);
  var0[1] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko]);
  C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
  __m256 var0_1[2];
  __m256 var1_1[2];
  var1_1[0] = _mm256_loadu_ps(&B.data[(4 * ko + 1) * (B.strides[0])]);
  var1_1[1] = _mm256_loadu_ps(&B.data[(4 * ko + 1) * (B.strides[0]) + 8]);
  var0_1[0] = _mm256_broadcast_ss(&A.data[4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_ss(&A.data[4 * ko + 1]);
  C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
  var0_1[0] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 1]);
  C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
  var0_1[0] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 1]);
  C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
  var0_1[0] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 1]);
  C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
  __m256 var0_2[2];
  __m256 var1_2[2];
  var1_2[0] = _mm256_loadu_ps(&B.data[(4 * ko + 2) * (B.strides[0])]);
  var1_2[1] = _mm256_loadu_ps(&B.data[(4 * ko + 2) * (B.strides[0]) + 8]);
  var0_2[0] = _mm256_broadcast_ss(&A.data[4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_ss(&A.data[4 * ko + 2]);
  C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
  var0_2[0] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 2]);
  C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
  var0_2[0] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 2]);
  C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
  var0_2[0] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 2]);
  C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
  __m256 var0_3[2];
  __m256 var1_3[2];
  var1_3[0] = _mm256_loadu_ps(&B.data[(4 * ko + 3) * (B.strides[0])]);
  var1_3[1] = _mm256_loadu_ps(&B.data[(4 * ko + 3) * (B.strides[0]) + 8]);
  var0_3[0] = _mm256_broadcast_ss(&A.data[4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_ss(&A.data[4 * ko + 3]);
  C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
  var0_3[0] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_ss(&A.data[A.strides[0] + 4 * ko + 3]);
  C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
  var0_3[0] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_ss(&A.data[(2) * (A.strides[0]) + 4 * ko + 3]);
  C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
  var0_3[0] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_ss(&A.data[(3) * (A.strides[0]) + 4 * ko + 3]);
  C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
}
_mm256_storeu_ps(&C.data[0], C_reg[0][0]);
_mm256_storeu_ps(&C.data[8], C_reg[0][1]);
_mm256_storeu_ps(&C.data[C.strides[0]], C_reg[1][0]);
_mm256_storeu_ps(&C.data[C.strides[0] + 8], C_reg[1][1]);
_mm256_storeu_ps(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
_mm256_storeu_ps(&C.data[(2) * (C.strides[0]) + 8], C_reg[2][1]);
_mm256_storeu_ps(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
_mm256_storeu_ps(&C.data[(3) * (C.strides[0]) + 8], C_reg[3][1]);
}

// avx2_microkernel_4x8_3(
//     C : [f64][4, 8] @DRAM,
//     A : [f64][4, 256] @DRAM,
//     B : [f64][256, 8] @DRAM
// )
static void avx2_microkernel_4x8_3( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B ) {
// assert stride(C, 1) == 1
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
__m256d C_reg[4][2];
C_reg[0][0] = _mm256_loadu_pd(&C.data[0]);
C_reg[0][1] = _mm256_loadu_pd(&C.data[4]);
C_reg[1][0] = _mm256_loadu_pd(&C.data[C.strides[0]]);
C_reg[1][1] = _mm256_loadu_pd(&C.data[C.strides[0] + 4]);
C_reg[2][0] = _mm256_loadu_pd(&C.data[(2) * (C.strides[0])]);
C_reg[2][1] = _mm256_loadu_pd(&C.data[(2) * (C.strides[0]) + 4]);
C_reg[3][0] = _mm256_loadu_pd(&C.data[(3) * (C.strides[0])]);
C_reg[3][1] = _mm256_loadu_pd(&C.data[(3) * (C.strides[0]) + 4]);
for (int_fast32_t ko = 0; ko < 64; ko++) {
  __m256d var0[2];
  __m256d var1[2];
  var1[0] = _mm256_loadu_pd(&B.data[(4 * ko) * (B.strides[0])]);
  var1[1] = _mm256_loadu_pd(&B.data[(4 * ko) * (B.strides[0]) + 4]);
  var0[0] = _mm256_broadcast_sd(&A.data[4 * ko]);
  var0[1] = _mm256_broadcast_sd(&A.data[4 * ko]);
  C_reg[0][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[0][1]);
  var0[0] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko]);
  var0[1] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko]);
  C_reg[1][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[1][1]);
  var0[0] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko]);
  var0[1] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko]);
  C_reg[2][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[2][1]);
  var0[0] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko]);
  var0[1] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko]);
  C_reg[3][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[3][1]);
  __m256d var0_1[2];
  __m256d var1_1[2];
  var1_1[0] = _mm256_loadu_pd(&B.data[(4 * ko + 1) * (B.strides[0])]);
  var1_1[1] = _mm256_loadu_pd(&B.data[(4 * ko + 1) * (B.strides[0]) + 4]);
  var0_1[0] = _mm256_broadcast_sd(&A.data[4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_sd(&A.data[4 * ko + 1]);
  C_reg[0][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[0][1]);
  var0_1[0] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 1]);
  C_reg[1][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[1][1]);
  var0_1[0] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 1]);
  C_reg[2][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[2][1]);
  var0_1[0] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 1]);
  var0_1[1] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 1]);
  C_reg[3][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[3][1]);
  __m256d var0_2[2];
  __m256d var1_2[2];
  var1_2[0] = _mm256_loadu_pd(&B.data[(4 * ko + 2) * (B.strides[0])]);
  var1_2[1] = _mm256_loadu_pd(&B.data[(4 * ko + 2) * (B.strides[0]) + 4]);
  var0_2[0] = _mm256_broadcast_sd(&A.data[4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_sd(&A.data[4 * ko + 2]);
  C_reg[0][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[0][1]);
  var0_2[0] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 2]);
  C_reg[1][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[1][1]);
  var0_2[0] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 2]);
  C_reg[2][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[2][1]);
  var0_2[0] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 2]);
  var0_2[1] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 2]);
  C_reg[3][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[3][1]);
  __m256d var0_3[2];
  __m256d var1_3[2];
  var1_3[0] = _mm256_loadu_pd(&B.data[(4 * ko + 3) * (B.strides[0])]);
  var1_3[1] = _mm256_loadu_pd(&B.data[(4 * ko + 3) * (B.strides[0]) + 4]);
  var0_3[0] = _mm256_broadcast_sd(&A.data[4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_sd(&A.data[4 * ko + 3]);
  C_reg[0][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[0][0]);
  C_reg[0][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[0][1]);
  var0_3[0] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_sd(&A.data[A.strides[0] + 4 * ko + 3]);
  C_reg[1][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[1][0]);
  C_reg[1][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[1][1]);
  var0_3[0] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_sd(&A.data[(2) * (A.strides[0]) + 4 * ko + 3]);
  C_reg[2][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[2][0]);
  C_reg[2][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[2][1]);
  var0_3[0] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 3]);
  var0_3[1] = _mm256_broadcast_sd(&A.data[(3) * (A.strides[0]) + 4 * ko + 3]);
  C_reg[3][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[3][0]);
  C_reg[3][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[3][1]);
}
_mm256_storeu_pd(&C.data[0], C_reg[0][0]);
_mm256_storeu_pd(&C.data[4], C_reg[0][1]);
_mm256_storeu_pd(&C.data[C.strides[0]], C_reg[1][0]);
_mm256_storeu_pd(&C.data[C.strides[0] + 4], C_reg[1][1]);
_mm256_storeu_pd(&C.data[(2) * (C.strides[0])], C_reg[2][0]);
_mm256_storeu_pd(&C.data[(2) * (C.strides[0]) + 4], C_reg[2][1]);
_mm256_storeu_pd(&C.data[(3) * (C.strides[0])], C_reg[3][0]);
_mm256_storeu_pd(&C.data[(3) * (C.strides[0]) + 4], C_reg[3][1]);
}


/* relying on the following instruction..."
avx2_reg_copy_pd(dst,src)
{dst_data} = {src_data};
*/

/* relying on the following instruction..."
avx2_reg_copy_ps(dst,src)
{dst_data} = {src_data};
*/
// d_apply_beta_128_256(
//     M : size,
//     N : size,
//     scalar : f64[1] @DRAM,
//     P : f64[M, M] @DRAM
// )
static void d_apply_beta_128_256( void *ctxt, int_fast32_t M, int_fast32_t N, const double* scalar, double* P ) {
for (int_fast32_t i = 0; i < M; i++) {
  __m256d scalar_vec;
  __m256d P_vec;
  __m256d P_vec2;
  for (int_fast32_t jo = 0; jo < ((1 + i) / (4)); jo++) {
    scalar_vec = _mm256_broadcast_sd(&scalar[0]);
    P_vec = _mm256_loadu_pd(&P[(i) * M + 4 * jo]);
    P_vec2 = P_vec;
    P_vec = _mm256_mul_pd(P_vec2, scalar_vec);
    _mm256_storeu_pd(&P[(i) * M + 4 * jo], P_vec);
  }
  if ((1 + i) % 4 > 0) {
    for (int_fast32_t ji = 0; ji < (1 + i) % 4; ji++) {
      P[i * M + ji + ((1 + i) / 4) * 4] = P[i * M + ji + ((1 + i) / 4) * 4] * scalar[0];
    }
  }
}
}

// d_diag_handler_scheduled(
//     A1 : [f64][128, 256] @DRAM,
//     A2 : [f64][256, 128] @DRAM,
//     C : [f64][128, 128] @DRAM
// )
static void d_diag_handler_scheduled( void *ctxt, struct exo_win_2f64c A1, struct exo_win_2f64c A2, struct exo_win_2f64 C ) {
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t io = 0; io < 4; io++) {
  for (int_fast32_t jo = 0; jo < io; jo++) {
    gebp_32x256_4(ctxt,(struct exo_win_2f64){ &C.data[(32 * io) * (C.strides[0]) + 32 * jo], { C.strides[0], 1 } },(struct exo_win_2f64c){ &A1.data[(32 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f64c){ &A2.data[32 * jo], { A2.strides[0], 1 } });
  }
  for (int_fast32_t iio = 0; iio < 8; iio++) {
    for (int_fast32_t jio = 0; jio < ((iio) / (2)); jio++) {
      avx2_microkernel_4x8_3(ctxt,(struct exo_win_2f64){ &C.data[(4 * iio + 32 * io) * (C.strides[0]) + 8 * jio + 32 * io], { C.strides[0], 1 } },(struct exo_win_2f64c){ &A1.data[(4 * iio + 32 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f64c){ &A2.data[8 * jio + 32 * io], { A2.strides[0], 1 } });
    }
    for (int_fast32_t iii = 0; iii < 4; iii++) {
      for (int_fast32_t jii = 0; jii < (iii + 4 * iio) % 8; jii++) {
        for (int_fast32_t k = 0; k < 256; k++) {
          C.data[(iii + 4 * iio + 32 * io) * C.strides[0] + jii + (iio / 2) * 8 + 32 * io] += A1.data[(iii + 4 * iio + 32 * io) * A1.strides[0] + k] * A2.data[k * A2.strides[0] + jii + (iio / 2) * 8 + 32 * io];
        }
      }
    }
  }
}
}

// dexo_dsyrk_lower_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C ) {
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    double *temp = (double*) malloc(1 * sizeof(*temp));
    temp[0] = 0.0;
    for (int_fast32_t k = 0; k < K; k++) {
      temp[0] += A1[i * K + k] * A2[j * K + k];
    }
    C[i * N + j] += alpha[0] * temp[0];
    free(temp);
  }
}
}

// dexo_dsyrk_lower_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(K >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t ko = 0; ko < ((K) / (256)); ko++) {
  gepp_dsyrk_scheduled(ctxt,N,(struct exo_win_2f64c){ &A1[256 * ko], { K, 1 } },(struct exo_win_2f64c){ &A2[(256 * ko) * N], { N, 1 } },(struct exo_win_2f64){ &C[0], { N, 1 } });
}
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    if (K % 256 > 0) {
      for (int_fast32_t ki = 0; ki < K % 256; ki++) {
        C[i * N + j] += A1[i * K + ki + (K / 256) * 256] * A2[(ki + (K / 256) * 256) * N + j];
      }
    }
  }
}
}

// dexo_dsyrk_lower_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C ) {
EXO_ASSUME(N == K);
double *temp = (double*) malloc(K * N * sizeof(*temp));
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    temp[j * N + k] = A1[j * N + k] * alpha[0];
  }
}
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * N + j] += temp[k * N + i] * A2[k * N + j];
    }
  }
}
free(temp);
}

// dexo_dsyrk_lower_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_lower_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(K >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
EXO_ASSUME(N == K);
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j];
    }
  }
}
}

// dexo_dsyrk_upper_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C ) {
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += alpha[0] * A1[i * K + k] * A2[j * K + k];
    }
  }
}
}

// dexo_dsyrk_upper_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C ) {
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[i * K + k] * A2[j * K + k];
    }
  }
}
}

// dexo_dsyrk_upper_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     alpha : f64[1] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* alpha, const double* A2, double* C ) {
EXO_ASSUME(K == N);
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j] * alpha[0];
    }
  }
}
}

// dexo_dsyrk_upper_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     C : f64[N, N] @DRAM
// )
static void dexo_dsyrk_upper_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const double* A1, const double* A2, double* C ) {
EXO_ASSUME(K == N);
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j];
    }
  }
}
}

// dsyrk_apply_scalar_upper(
//     M : size,
//     N : size,
//     scalar : f64[1] @DRAM,
//     P : f64[M, M] @DRAM
// )
static void dsyrk_apply_scalar_upper( void *ctxt, int_fast32_t M, int_fast32_t N, const double* scalar, double* P ) {
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < M - i; j++) {
    P[i * M + j] = P[i * M + j] * scalar[0];
  }
}
}

// exo_dsyrk_lower_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
d_apply_beta_128_256(ctxt,N,N,beta,C);
}

// exo_dsyrk_lower_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
d_apply_beta_128_256(ctxt,N,N,beta,C);
dexo_dsyrk_lower_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_lower_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dexo_dsyrk_lower_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_lower_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dexo_dsyrk_lower_notranspose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_dsyrk_lower_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(K == N);
d_apply_beta_128_256(ctxt,N,N,beta,C);
dexo_dsyrk_lower_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_lower_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(K == N);
dexo_dsyrk_lower_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_lower_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_lower_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(K == N);
dexo_dsyrk_lower_transpose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_dsyrk_upper_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dsyrk_apply_scalar_upper(ctxt,N,N,beta,C);
}

// exo_dsyrk_upper_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dsyrk_apply_scalar_upper(ctxt,N,N,beta,C);
dexo_dsyrk_upper_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_upper_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dexo_dsyrk_upper_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_upper_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[N, K] @DRAM,
//     A2 : f64[N, K] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
dexo_dsyrk_upper_notranspose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_dsyrk_upper_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(N == K);
dsyrk_apply_scalar_upper(ctxt,N,N,beta,C);
dexo_dsyrk_upper_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_upper_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(N == K);
dexo_dsyrk_upper_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_dsyrk_upper_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f64[1] @DRAM,
//     A1 : f64[K, N] @DRAM,
//     A2 : f64[K, N] @DRAM,
//     beta : f64[1] @DRAM,
//     C : f64[N, N] @DRAM
// )
void exo_dsyrk_upper_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const double* alpha, const double* A1, const double* A2, const double* beta, double* C ) {
EXO_ASSUME(N == K);
dexo_dsyrk_upper_transpose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_ssyrk_lower_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
s_apply_beta_128_256(ctxt,N,N,beta,C);
}

// exo_ssyrk_lower_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
s_apply_beta_128_256(ctxt,N,N,beta,C);
sexo_ssyrk_lower_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_lower_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
sexo_ssyrk_lower_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_lower_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
sexo_ssyrk_lower_notranspose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_ssyrk_lower_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(K == N);
s_apply_beta_128_256(ctxt,N,N,beta,C);
sexo_ssyrk_lower_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_lower_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(K == N);
sexo_ssyrk_lower_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_lower_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_lower_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(K == N);
sexo_ssyrk_lower_transpose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_ssyrk_upper_alphazero_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_alphazero_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
ssyrk_apply_scalar_upper(ctxt,N,N,beta,C);
}

// exo_ssyrk_upper_notranspose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
ssyrk_apply_scalar_upper(ctxt,N,N,beta,C);
sexo_ssyrk_upper_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_upper_notranspose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
sexo_ssyrk_upper_notranspose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_upper_notranspose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_notranspose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
sexo_ssyrk_upper_notranspose_noalpha(ctxt,N,K,A1,A2,C);
}

// exo_ssyrk_upper_transpose_alpha_beta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_alpha_beta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(N == K);
ssyrk_apply_scalar_upper(ctxt,N,N,beta,C);
sexo_ssyrk_upper_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_upper_transpose_alpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_alpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(N == K);
sexo_ssyrk_upper_transpose_alpha(ctxt,N,K,A1,alpha,A2,C);
}

// exo_ssyrk_upper_transpose_noalpha_nobeta(
//     N : size,
//     K : size,
//     alpha : f32[1] @DRAM,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[N, N] @DRAM
// )
void exo_ssyrk_upper_transpose_noalpha_nobeta( void *ctxt, int_fast32_t N, int_fast32_t K, const float* alpha, const float* A1, const float* A2, const float* beta, float* C ) {
EXO_ASSUME(N == K);
sexo_ssyrk_upper_transpose_noalpha(ctxt,N,K,A1,A2,C);
}

// gebp_128x256_0(
//     C : [f32][128, 128] @DRAM,
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM
// )
static void gebp_128x256_0( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t jo = 0; jo < 8; jo++) {
  float *B_reg_strip = (float*) malloc(256 * 16 * sizeof(*B_reg_strip));
  for (int_fast32_t i0 = 0; i0 < 256; i0++) {
    for (int_fast32_t i1 = 0; i1 < 16; i1++) {
      B_reg_strip[i0 * 16 + i1] = B.data[i0 * B.strides[0] + i1 + 16 * jo];
    }
  }
  for (int_fast32_t io = 0; io < 32; io++) {
    __m256 C_reg[4][2];
    C_reg[0][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[0][1] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[1][0] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[1][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[2][0] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[2][1] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[3][0] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[3][1] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    for (int_fast32_t ko = 0; ko < 64; ko++) {
      __m256 var0[2];
      __m256 var1[2];
      var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16)]);
      var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16) + 8]);
      var0[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
      __m256 var0_1[2];
      __m256 var1_1[2];
      var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16)]);
      var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16) + 8]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
      __m256 var0_2[2];
      __m256 var1_2[2];
      var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16)]);
      var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16) + 8]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
      __m256 var0_3[2];
      __m256 var1_3[2];
      var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16)]);
      var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16) + 8]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
    }
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo], C_reg[0][0]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[0][1]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[1][0]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[1][1]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[2][0]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[2][1]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[3][0]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[3][1]);
  }
  free(B_reg_strip);
}
}

// gebp_128x256_2(
//     C : [f32][128, 128] @DRAM,
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM
// )
static void gebp_128x256_2( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t jo = 0; jo < 8; jo++) {
  float *B_reg_strip = (float*) malloc(256 * 16 * sizeof(*B_reg_strip));
  for (int_fast32_t i0 = 0; i0 < 256; i0++) {
    for (int_fast32_t i1 = 0; i1 < 16; i1++) {
      B_reg_strip[i0 * 16 + i1] = B.data[i0 * B.strides[0] + i1 + 16 * jo];
    }
  }
  for (int_fast32_t io = 0; io < 32; io++) {
    __m256 C_reg[4][2];
    C_reg[0][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[0][1] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[1][0] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[1][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[2][0] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[2][1] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[3][0] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[3][1] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    for (int_fast32_t ko = 0; ko < 64; ko++) {
      __m256 var0[2];
      __m256 var1[2];
      var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16)]);
      var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16) + 8]);
      var0[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
      __m256 var0_1[2];
      __m256 var1_1[2];
      var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16)]);
      var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16) + 8]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
      __m256 var0_2[2];
      __m256 var1_2[2];
      var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16)]);
      var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16) + 8]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
      __m256 var0_3[2];
      __m256 var1_3[2];
      var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16)]);
      var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16) + 8]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
    }
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo], C_reg[0][0]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[0][1]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[1][0]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[1][1]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[2][0]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[2][1]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[3][0]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[3][1]);
  }
  free(B_reg_strip);
}
}

// gebp_128x256_3(
//     C : [f64][128, 128] @DRAM,
//     A : [f64][128, 256] @DRAM,
//     B : [f64][256, 128] @DRAM
// )
static void gebp_128x256_3( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t jo = 0; jo < 16; jo++) {
  double *B_reg_strip = (double*) malloc(256 * 8 * sizeof(*B_reg_strip));
  for (int_fast32_t i0 = 0; i0 < 256; i0++) {
    for (int_fast32_t i1 = 0; i1 < 8; i1++) {
      B_reg_strip[i0 * 8 + i1] = B.data[i0 * B.strides[0] + i1 + 8 * jo];
    }
  }
  for (int_fast32_t io = 0; io < 32; io++) {
    __m256d C_reg[4][2];
    C_reg[0][0] = _mm256_loadu_pd(&C.data[(4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[0][1] = _mm256_loadu_pd(&C.data[(4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[1][0] = _mm256_loadu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[1][1] = _mm256_loadu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[2][0] = _mm256_loadu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[2][1] = _mm256_loadu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[3][0] = _mm256_loadu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[3][1] = _mm256_loadu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    for (int_fast32_t ko = 0; ko < 64; ko++) {
      __m256d var0[2];
      __m256d var1[2];
      var1[0] = _mm256_loadu_pd(&B_reg_strip[(4 * ko) * 8]);
      var1[1] = _mm256_loadu_pd(&B_reg_strip[(4 * ko) * 8 + 4]);
      var0[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[0][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[1][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[2][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[3][1]);
      __m256d var0_1[2];
      __m256d var1_1[2];
      var1_1[0] = _mm256_loadu_pd(&B_reg_strip[(1 + 4 * ko) * 8]);
      var1_1[1] = _mm256_loadu_pd(&B_reg_strip[(1 + 4 * ko) * 8 + 4]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[0][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[1][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[2][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[3][1]);
      __m256d var0_2[2];
      __m256d var1_2[2];
      var1_2[0] = _mm256_loadu_pd(&B_reg_strip[(2 + 4 * ko) * 8]);
      var1_2[1] = _mm256_loadu_pd(&B_reg_strip[(2 + 4 * ko) * 8 + 4]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[0][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[1][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[2][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[3][1]);
      __m256d var0_3[2];
      __m256d var1_3[2];
      var1_3[0] = _mm256_loadu_pd(&B_reg_strip[(3 + 4 * ko) * 8]);
      var1_3[1] = _mm256_loadu_pd(&B_reg_strip[(3 + 4 * ko) * 8 + 4]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[0][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[1][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[2][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[3][1]);
    }
    _mm256_storeu_pd(&C.data[(4 * io) * (C.strides[0]) + 8 * jo], C_reg[0][0]);
    _mm256_storeu_pd(&C.data[(4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[0][1]);
    _mm256_storeu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[1][0]);
    _mm256_storeu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[1][1]);
    _mm256_storeu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[2][0]);
    _mm256_storeu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[2][1]);
    _mm256_storeu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[3][0]);
    _mm256_storeu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[3][1]);
  }
  free(B_reg_strip);
}
}

// gebp_32x256_1(
//     C : [f32][32, 32] @DRAM,
//     A : [f32][32, 256] @DRAM,
//     B : [f32][256, 32] @DRAM
// )
static void gebp_32x256_1( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t jo = 0; jo < 2; jo++) {
  float *B_reg_strip = (float*) malloc(256 * 16 * sizeof(*B_reg_strip));
  for (int_fast32_t i0 = 0; i0 < 256; i0++) {
    for (int_fast32_t i1 = 0; i1 < 16; i1++) {
      B_reg_strip[i0 * 16 + i1] = B.data[i0 * B.strides[0] + i1 + 16 * jo];
    }
  }
  for (int_fast32_t io = 0; io < 8; io++) {
    __m256 C_reg[4][2];
    C_reg[0][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[0][1] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[1][0] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[1][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[2][0] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[2][1] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    C_reg[3][0] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo]);
    C_reg[3][1] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo]);
    for (int_fast32_t ko = 0; ko < 64; ko++) {
      __m256 var0[2];
      __m256 var1[2];
      var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16)]);
      var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko) * (16) + 8]);
      var0[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
      var0[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
      __m256 var0_1[2];
      __m256 var1_1[2];
      var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16)]);
      var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko) * (16) + 8]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
      var0_1[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
      __m256 var0_2[2];
      __m256 var1_2[2];
      var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16)]);
      var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko) * (16) + 8]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
      var0_2[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
      __m256 var0_3[2];
      __m256 var1_3[2];
      var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16)]);
      var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko) * (16) + 8]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
      var0_3[0] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_ss(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
    }
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 * jo], C_reg[0][0]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[0][1]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[1][0]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[1][1]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[2][0]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[2][1]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 * jo], C_reg[3][0]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 16 * jo], C_reg[3][1]);
  }
  free(B_reg_strip);
}
}

// gebp_32x256_4(
//     C : [f64][32, 32] @DRAM,
//     A : [f64][32, 256] @DRAM,
//     B : [f64][256, 32] @DRAM
// )
static void gebp_32x256_4( void *ctxt, struct exo_win_2f64 C, struct exo_win_2f64c A, struct exo_win_2f64c B ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t jo = 0; jo < 4; jo++) {
  double *B_reg_strip = (double*) malloc(256 * 8 * sizeof(*B_reg_strip));
  for (int_fast32_t i0 = 0; i0 < 256; i0++) {
    for (int_fast32_t i1 = 0; i1 < 8; i1++) {
      B_reg_strip[i0 * 8 + i1] = B.data[i0 * B.strides[0] + i1 + 8 * jo];
    }
  }
  for (int_fast32_t io = 0; io < 8; io++) {
    __m256d C_reg[4][2];
    C_reg[0][0] = _mm256_loadu_pd(&C.data[(4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[0][1] = _mm256_loadu_pd(&C.data[(4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[1][0] = _mm256_loadu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[1][1] = _mm256_loadu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[2][0] = _mm256_loadu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[2][1] = _mm256_loadu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    C_reg[3][0] = _mm256_loadu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 * jo]);
    C_reg[3][1] = _mm256_loadu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 4 + 8 * jo]);
    for (int_fast32_t ko = 0; ko < 64; ko++) {
      __m256d var0[2];
      __m256d var1[2];
      var1[0] = _mm256_loadu_pd(&B_reg_strip[(4 * ko) * 8]);
      var1[1] = _mm256_loadu_pd(&B_reg_strip[(4 * ko) * 8 + 4]);
      var0[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[0][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[1][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[2][1]);
      var0[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      var0[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0[0], var1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0[1], var1[1], C_reg[3][1]);
      __m256d var0_1[2];
      __m256d var1_1[2];
      var1_1[0] = _mm256_loadu_pd(&B_reg_strip[(1 + 4 * ko) * 8]);
      var1_1[1] = _mm256_loadu_pd(&B_reg_strip[(1 + 4 * ko) * 8 + 4]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[0][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[1][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[2][1]);
      var0_1[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      var0_1[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 1 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_1[0], var1_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_1[1], var1_1[1], C_reg[3][1]);
      __m256d var0_2[2];
      __m256d var1_2[2];
      var1_2[0] = _mm256_loadu_pd(&B_reg_strip[(2 + 4 * ko) * 8]);
      var1_2[1] = _mm256_loadu_pd(&B_reg_strip[(2 + 4 * ko) * 8 + 4]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[0][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[1][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[2][1]);
      var0_2[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      var0_2[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 2 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_2[0], var1_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_2[1], var1_2[1], C_reg[3][1]);
      __m256d var0_3[2];
      __m256d var1_3[2];
      var1_3[0] = _mm256_loadu_pd(&B_reg_strip[(3 + 4 * ko) * 8]);
      var1_3[1] = _mm256_loadu_pd(&B_reg_strip[(3 + 4 * ko) * 8 + 4]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[0][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[0][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(1 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[1][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[1][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(2 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[2][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[2][1]);
      var0_3[0] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      var0_3[1] = _mm256_broadcast_sd(&A.data[(3 + 4 * io) * (A.strides[0]) + 3 + 4 * ko]);
      C_reg[3][0] = _mm256_fmadd_pd(var0_3[0], var1_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_pd(var0_3[1], var1_3[1], C_reg[3][1]);
    }
    _mm256_storeu_pd(&C.data[(4 * io) * (C.strides[0]) + 8 * jo], C_reg[0][0]);
    _mm256_storeu_pd(&C.data[(4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[0][1]);
    _mm256_storeu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[1][0]);
    _mm256_storeu_pd(&C.data[(1 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[1][1]);
    _mm256_storeu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[2][0]);
    _mm256_storeu_pd(&C.data[(2 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[2][1]);
    _mm256_storeu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 * jo], C_reg[3][0]);
    _mm256_storeu_pd(&C.data[(3 + 4 * io) * (C.strides[0]) + 4 + 8 * jo], C_reg[3][1]);
  }
  free(B_reg_strip);
}
}

// gepp_dsyrk_scheduled(
//     N : size,
//     A1 : [f64][N, 256] @DRAM,
//     A2 : [f64][256, N] @DRAM,
//     C : [f64][N, N] @DRAM
// )
static void gepp_dsyrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f64c A1, struct exo_win_2f64c A2, struct exo_win_2f64 C ) {
EXO_ASSUME(N >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t io = 0; io < ((N) / (128)); io++) {
  for (int_fast32_t jo = 0; jo < io; jo++) {
    gebp_128x256_3(ctxt,(struct exo_win_2f64){ &C.data[(128 * io) * (C.strides[0]) + 1 + 128 * jo], { C.strides[0], 1 } },(struct exo_win_2f64c){ &A1.data[(128 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f64c){ &A2.data[1 + 128 * jo], { A2.strides[0], 1 } });
  }
}
for (int_fast32_t io = 0; io < ((N) / (128)); io++) {
  for (int_fast32_t ii = 0; ii < 128; ii++) {
    for (int_fast32_t j = 0; j < 1; j++) {
      for (int_fast32_t k = 0; k < 256; k++) {
        C.data[(ii + 128 * io) * C.strides[0] + j] += A1.data[(ii + 128 * io) * A1.strides[0] + k] * A2.data[k * A2.strides[0] + j];
      }
    }
  }
  d_diag_handler_scheduled(ctxt,(struct exo_win_2f64c){ &A1.data[(128 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f64c){ &A2.data[1 + 128 * io], { A2.strides[0], 1 } },(struct exo_win_2f64){ &C.data[(128 * io) * (C.strides[0]) + 1 + 128 * io], { C.strides[0], 1 } });
}
for (int_fast32_t ii = 0; ii < N % 128; ii++) {
  for (int_fast32_t j = 0; j < 1 + (ii + ((N) / (128)) * 128); j++) {
    for (int_fast32_t k = 0; k < 256; k++) {
      C.data[(ii + (N / 128) * 128) * C.strides[0] + j] += A1.data[(ii + (N / 128) * 128) * A1.strides[0] + k] * A2.data[k * A2.strides[0] + j];
    }
  }
}
}

// gepp_ssyrk_scheduled(
//     N : size,
//     A1 : [f32][N, 256] @DRAM,
//     A2 : [f32][256, N] @DRAM,
//     C : [f32][N, N] @DRAM
// )
static void gepp_ssyrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C ) {
EXO_ASSUME(N >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t io = 0; io < ((N) / (128)); io++) {
  for (int_fast32_t jo = 0; jo < io; jo++) {
    gebp_128x256_0(ctxt,(struct exo_win_2f32){ &C.data[(128 * io) * (C.strides[0]) + 1 + 128 * jo], { C.strides[0], 1 } },(struct exo_win_2f32c){ &A1.data[(128 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f32c){ &A2.data[1 + 128 * jo], { A2.strides[0], 1 } });
  }
}
for (int_fast32_t io = 0; io < ((N) / (128)); io++) {
  for (int_fast32_t ii = 0; ii < 128; ii++) {
    for (int_fast32_t j = 0; j < 1; j++) {
      for (int_fast32_t k = 0; k < 256; k++) {
        C.data[(ii + 128 * io) * C.strides[0] + j] += A1.data[(ii + 128 * io) * A1.strides[0] + k] * A2.data[k * A2.strides[0] + j];
      }
    }
  }
  s_diag_handler_scheduled(ctxt,(struct exo_win_2f32c){ &A1.data[(128 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f32c){ &A2.data[1 + 128 * io], { A2.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(128 * io) * (C.strides[0]) + 1 + 128 * io], { C.strides[0], 1 } });
}
for (int_fast32_t ii = 0; ii < N % 128; ii++) {
  for (int_fast32_t j = 0; j < 1 + (ii + ((N) / (128)) * 128); j++) {
    for (int_fast32_t k = 0; k < 256; k++) {
      C.data[(ii + (N / 128) * 128) * C.strides[0] + j] += A1.data[(ii + (N / 128) * 128) * A1.strides[0] + k] * A2.data[k * A2.strides[0] + j];
    }
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
mm256_mul_pd(out,x,y)
{out_data} = _mm256_mul_pd({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_mul_ps(out,x,y)
{out_data} = _mm256_mul_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_storeu_pd(dst,src)
_mm256_storeu_pd(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
// s_apply_beta_128_256(
//     M : size,
//     N : size,
//     scalar : f32[1] @DRAM,
//     P : f32[M, M] @DRAM
// )
static void s_apply_beta_128_256( void *ctxt, int_fast32_t M, int_fast32_t N, const float* scalar, float* P ) {
for (int_fast32_t i = 0; i < M; i++) {
  __m256 scalar_vec;
  __m256 P_vec;
  __m256 P_vec2;
  for (int_fast32_t jo = 0; jo < ((1 + i) / (8)); jo++) {
    scalar_vec = _mm256_broadcast_ss(&scalar[0]);
    P_vec = _mm256_loadu_ps(&P[(i) * M + 8 * jo]);
    P_vec2 = P_vec;
    P_vec = _mm256_mul_ps(P_vec2, scalar_vec);
    _mm256_storeu_ps(&P[(i) * M + 8 * jo], P_vec);
  }
  if ((1 + i) % 8 > 0) {
    for (int_fast32_t ji = 0; ji < (1 + i) % 8; ji++) {
      P[i * M + ji + ((1 + i) / 8) * 8] = P[i * M + ji + ((1 + i) / 8) * 8] * scalar[0];
    }
  }
}
}

// s_diag_handler_scheduled(
//     A1 : [f32][128, 256] @DRAM,
//     A2 : [f32][256, 128] @DRAM,
//     C : [f32][128, 128] @DRAM
// )
static void s_diag_handler_scheduled( void *ctxt, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C ) {
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t io = 0; io < 4; io++) {
  for (int_fast32_t jo = 0; jo < io; jo++) {
    gebp_32x256_1(ctxt,(struct exo_win_2f32){ &C.data[(32 * io) * (C.strides[0]) + 32 * jo], { C.strides[0], 1 } },(struct exo_win_2f32c){ &A1.data[(32 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f32c){ &A2.data[32 * jo], { A2.strides[0], 1 } });
  }
  for (int_fast32_t iio = 0; iio < 8; iio++) {
    for (int_fast32_t jio = 0; jio < ((iio) / (4)); jio++) {
      avx2_microkernel_4x16_1(ctxt,(struct exo_win_2f32){ &C.data[(4 * iio + 32 * io) * (C.strides[0]) + 16 * jio + 32 * io], { C.strides[0], 1 } },(struct exo_win_2f32c){ &A1.data[(4 * iio + 32 * io) * (A1.strides[0])], { A1.strides[0], 1 } },(struct exo_win_2f32c){ &A2.data[16 * jio + 32 * io], { A2.strides[0], 1 } });
    }
  }
}
s_unsafe_microkernel_scheduled(ctxt,A1,A2,C);
}

// s_unsafe_microkernel_scheduled(
//     A : [f32][128, 256] @DRAM,
//     B : [f32][256, 128] @DRAM,
//     C : [f32][128, 128] @DRAM
// )
static void s_unsafe_microkernel_scheduled( void *ctxt, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C ) {
// assert stride(C, 1) == 1
// assert stride(B, 1) == 1
// assert stride(A, 1) == 1
float *C_reg = (float*) malloc(128 * 128 * sizeof(*C_reg));
for (int_fast32_t i = 0; i < 128; i++) {
  for (int_fast32_t j = 0; j < 128; j++) {
    C_reg[i * 128 + j] = 0.0;
  }
}
gebp_128x256_2(ctxt,(struct exo_win_2f32){ &C_reg[0], { 128, 1 } },(struct exo_win_2f32c){ &A.data[0], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[0], { B.strides[0], 1 } });
for (int_fast32_t i = 0; i < 128; i++) {
  for (int_fast32_t j = 0; j < i % 16; j++) {
    C.data[i * C.strides[0] + j + (i / 16) * 16] += C_reg[i * 128 + j + (i / 16) * 16];
  }
}
free(C_reg);
}

// sexo_ssyrk_lower_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C ) {
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    float *temp = (float*) malloc(1 * sizeof(*temp));
    temp[0] = 0.0;
    for (int_fast32_t k = 0; k < K; k++) {
      temp[0] += A1[i * K + k] * A2[j * K + k];
    }
    C[i * N + j] += alpha[0] * temp[0];
    free(temp);
  }
}
}

// sexo_ssyrk_lower_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(K >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t ko = 0; ko < ((K) / (256)); ko++) {
  gepp_ssyrk_scheduled(ctxt,N,(struct exo_win_2f32c){ &A1[256 * ko], { K, 1 } },(struct exo_win_2f32c){ &A2[(256 * ko) * N], { N, 1 } },(struct exo_win_2f32){ &C[0], { N, 1 } });
}
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    if (K % 256 > 0) {
      for (int_fast32_t ki = 0; ki < K % 256; ki++) {
        C[i * N + j] += A1[i * K + ki + (K / 256) * 256] * A2[(ki + (K / 256) * 256) * N + j];
      }
    }
  }
}
}

// sexo_ssyrk_lower_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C ) {
EXO_ASSUME(N == K);
float *temp = (float*) malloc(K * N * sizeof(*temp));
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    temp[j * N + k] = A1[j * N + k] * alpha[0];
  }
}
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * N + j] += temp[k * N + i] * A2[k * N + j];
    }
  }
}
free(temp);
}

// sexo_ssyrk_lower_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_lower_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(K >= 1);
// assert stride(A1, 1) == 1
// assert stride(A2, 1) == 1
// assert stride(C, 1) == 1
EXO_ASSUME(N == K);
for (int_fast32_t i = 0; i < N; i++) {
  for (int_fast32_t j = 0; j < 1 + i; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j];
    }
  }
}
}

// sexo_ssyrk_upper_notranspose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_notranspose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C ) {
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += alpha[0] * A1[i * K + k] * A2[j * K + k];
    }
  }
}
}

// sexo_ssyrk_upper_notranspose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[N, K] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_notranspose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C ) {
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[i * K + k] * A2[j * K + k];
    }
  }
}
}

// sexo_ssyrk_upper_transpose_alpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     alpha : f32[1] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_transpose_alpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* alpha, const float* A2, float* C ) {
EXO_ASSUME(K == N);
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j] * alpha[0];
    }
  }
}
}

// sexo_ssyrk_upper_transpose_noalpha(
//     N : size,
//     K : size,
//     A1 : f32[K, N] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
static void sexo_ssyrk_upper_transpose_noalpha( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C ) {
EXO_ASSUME(K == N);
for (int_fast32_t j = 0; j < N; j++) {
  for (int_fast32_t k = 0; k < K; k++) {
    for (int_fast32_t i = 0; i < 1 + j; i++) {
      C[i * N + j] += A1[k * N + i] * A2[k * N + j];
    }
  }
}
}

// ssyrk_apply_scalar_upper(
//     M : size,
//     N : size,
//     scalar : f32[1] @DRAM,
//     P : f32[M, M] @DRAM
// )
static void ssyrk_apply_scalar_upper( void *ctxt, int_fast32_t M, int_fast32_t N, const float* scalar, float* P ) {
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < M - i; j++) {
    P[i * M + j] = P[i * M + j] * scalar[0];
  }
}
}

