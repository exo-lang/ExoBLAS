#include "exo_symm.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>



// exo_ssymm_lower_left_noalpha_nobeta_192_192_192(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_192_192_192( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[192 * 16];
static float B_strip[192 * 192];
for (int_fast32_t jo = 0; jo < ((N) / (192)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (192)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 192; i0++) {
      for (int_fast32_t i1 = 0; i1 < 192; i1++) {
        B_strip[i0 * 192 + i1] = B[(i0 + 192 * ko) * N + i1 + 192 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (192)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 12; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 192; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 192 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 32; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 48; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 4 * ko_1 + 192 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 1 + 4 * ko_1 + 192 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 2 + 4 * ko_1 + 192 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 192 * io) * K + 3 + 4 * ko_1 + 192 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 192 * io) * N + 16 * jo_1 + 192 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 192 * io) * N + 8 + 16 * jo_1 + 192 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 192 > 0) {
      for (int_fast32_t ii = 0; ii < M % 192; ii++) {
        for (int_fast32_t j = 0; j < 192; j++) {
          for (int_fast32_t k = 0; k < 192; k++) {
            C[(ii + (M / 192) * 192) * N + j + 192 * jo] += A[(ii + (M / 192) * 192) * K + k + 192 * ko] * B_strip[k * 192 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (192)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 192; ji++) {
      for (int_fast32_t ki = 0; ki < 192; ki++) {
        C[i * N + ji + (N / 192) * 192] += A[i * K + ki + 192 * ko] * B[(ki + 192 * ko) * N + ji + (N / 192) * 192];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 192; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 192) * 192] * B[(ki + (K / 192) * 192) * N + j];
    }
  }
}
}

// exo_ssymm_lower_left_noalpha_nobeta_384_384_384(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_384_384_384( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[384 * 16];
static float B_strip[384 * 384];
for (int_fast32_t jo = 0; jo < ((N) / (384)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (384)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 384; i0++) {
      for (int_fast32_t i1 = 0; i1 < 384; i1++) {
        B_strip[i0 * 384 + i1] = B[(i0 + 384 * ko) * N + i1 + 384 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (384)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 24; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 384; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 384 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 64; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 96; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 4 * ko_1 + 384 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 1 + 4 * ko_1 + 384 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 2 + 4 * ko_1 + 384 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 384 * io) * K + 3 + 4 * ko_1 + 384 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 384 * io) * N + 16 * jo_1 + 384 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 384 * io) * N + 8 + 16 * jo_1 + 384 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 384 > 0) {
      for (int_fast32_t ii = 0; ii < M % 384; ii++) {
        for (int_fast32_t j = 0; j < 384; j++) {
          for (int_fast32_t k = 0; k < 384; k++) {
            C[(ii + (M / 384) * 384) * N + j + 384 * jo] += A[(ii + (M / 384) * 384) * K + k + 384 * ko] * B_strip[k * 384 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (384)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 384; ji++) {
      for (int_fast32_t ki = 0; ki < 384; ki++) {
        C[i * N + ji + (N / 384) * 384] += A[i * K + ki + 384 * ko] * B[(ki + 384 * ko) * N + ji + (N / 384) * 384];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 384; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 384) * 384] * B[(ki + (K / 384) * 384) * N + j];
    }
  }
}
}

// exo_ssymm_lower_left_noalpha_nobeta_480_240_480(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_480_240_480( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[480 * 16];
static float B_strip[480 * 480];
for (int_fast32_t jo = 0; jo < ((N) / (480)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (480)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 480; i0++) {
      for (int_fast32_t i1 = 0; i1 < 480; i1++) {
        B_strip[i0 * 480 + i1] = B[(i0 + 480 * ko) * N + i1 + 480 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (240)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 30; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 480; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 480 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 40; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 120; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 480 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 480 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 240 > 0) {
      for (int_fast32_t ii = 0; ii < M % 240; ii++) {
        for (int_fast32_t j = 0; j < 480; j++) {
          for (int_fast32_t k = 0; k < 480; k++) {
            C[(ii + (M / 240) * 240) * N + j + 480 * jo] += A[(ii + (M / 240) * 240) * K + k + 480 * ko] * B_strip[k * 480 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (480)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 480; ji++) {
      for (int_fast32_t ki = 0; ki < 480; ki++) {
        C[i * N + ji + (N / 480) * 480] += A[i * K + ki + 480 * ko] * B[(ki + 480 * ko) * N + ji + (N / 480) * 480];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 480; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 480) * 480] * B[(ki + (K / 480) * 480) * N + j];
    }
  }
}
}

// exo_ssymm_lower_left_noalpha_nobeta_48_48_48(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_48_48_48( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[48 * 16];
static float B_strip[48 * 48];
for (int_fast32_t jo = 0; jo < ((N) / (48)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (48)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 48; i0++) {
      for (int_fast32_t i1 = 0; i1 < 48; i1++) {
        B_strip[i0 * 48 + i1] = B[(i0 + 48 * ko) * N + i1 + 48 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (48)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 3; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 48; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 48 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 8; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 12; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 4 * ko_1 + 48 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 1 + 4 * ko_1 + 48 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 2 + 4 * ko_1 + 48 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 48 * io) * K + 3 + 4 * ko_1 + 48 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 48 * io) * N + 16 * jo_1 + 48 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 48 * io) * N + 8 + 16 * jo_1 + 48 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 48 > 0) {
      for (int_fast32_t ii = 0; ii < M % 48; ii++) {
        for (int_fast32_t j = 0; j < 48; j++) {
          for (int_fast32_t k = 0; k < 48; k++) {
            C[(ii + (M / 48) * 48) * N + j + 48 * jo] += A[(ii + (M / 48) * 48) * K + k + 48 * ko] * B_strip[k * 48 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (48)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 48; ji++) {
      for (int_fast32_t ki = 0; ki < 48; ki++) {
        C[i * N + ji + (N / 48) * 48] += A[i * K + ki + 48 * ko] * B[(ki + 48 * ko) * N + ji + (N / 48) * 48];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 48; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 48) * 48] * B[(ki + (K / 48) * 48) * N + j];
    }
  }
}
}

// exo_ssymm_lower_left_noalpha_nobeta_960_240_480(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_960_240_480( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[480 * 16];
static float B_strip[480 * 960];
for (int_fast32_t jo = 0; jo < ((N) / (960)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (480)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 480; i0++) {
      for (int_fast32_t i1 = 0; i1 < 960; i1++) {
        B_strip[i0 * 960 + i1] = B[(i0 + 480 * ko) * N + i1 + 960 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (240)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 60; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 480; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 960 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 40; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 120; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 1 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 2 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 240 * io) * K + 3 + 4 * ko_1 + 480 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 16 * jo_1 + 960 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 240 * io) * N + 8 + 16 * jo_1 + 960 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 240 > 0) {
      for (int_fast32_t ii = 0; ii < M % 240; ii++) {
        for (int_fast32_t j = 0; j < 960; j++) {
          for (int_fast32_t k = 0; k < 480; k++) {
            C[(ii + (M / 240) * 240) * N + j + 960 * jo] += A[(ii + (M / 240) * 240) * K + k + 480 * ko] * B_strip[k * 960 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (480)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 960; ji++) {
      for (int_fast32_t ki = 0; ki < 480; ki++) {
        C[i * N + ji + (N / 960) * 960] += A[i * K + ki + 480 * ko] * B[(ki + 480 * ko) * N + ji + (N / 960) * 960];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 480; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 480) * 480] * B[(ki + (K / 480) * 480) * N + j];
    }
  }
}
}

// exo_ssymm_lower_left_noalpha_nobeta_96_96_96(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void exo_ssymm_lower_left_noalpha_nobeta_96_96_96( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
static float B_reg_strip[96 * 16];
static float B_strip[96 * 96];
for (int_fast32_t jo = 0; jo < ((N) / (96)); jo++) {
  for (int_fast32_t ko = 0; ko < ((K) / (96)); ko++) {
    for (int_fast32_t i0 = 0; i0 < 96; i0++) {
      for (int_fast32_t i1 = 0; i1 < 96; i1++) {
        B_strip[i0 * 96 + i1] = B[(i0 + 96 * ko) * N + i1 + 96 * jo];
      }
    }
    for (int_fast32_t io = 0; io < ((M) / (96)); io++) {
      for (int_fast32_t jo_1 = 0; jo_1 < 6; jo_1++) {
        for (int_fast32_t i0 = 0; i0 < 96; i0++) {
          for (int_fast32_t i1 = 0; i1 < 16; i1++) {
            B_reg_strip[i0 * 16 + i1] = B_strip[i0 * 96 + i1 + 16 * jo_1];
          }
        }
        for (int_fast32_t io_1 = 0; io_1 < 16; io_1++) {
          __m256 C_reg[6][2];
          C_reg[0][0] = _mm256_loadu_ps(&C[(6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[0][1] = _mm256_loadu_ps(&C[(6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          C_reg[1][0] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[1][1] = _mm256_loadu_ps(&C[(1 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          C_reg[2][0] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[2][1] = _mm256_loadu_ps(&C[(2 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          C_reg[3][0] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[3][1] = _mm256_loadu_ps(&C[(3 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          C_reg[4][0] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[4][1] = _mm256_loadu_ps(&C[(4 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          C_reg[5][0] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo]);
          C_reg[5][1] = _mm256_loadu_ps(&C[(5 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo]);
          for (int_fast32_t ko_1 = 0; ko_1 < 24; ko_1++) {
            __m256 var0[2];
            __m256 var1[2];
            var1[0] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16)]);
            var1[1] = _mm256_loadu_ps(&B_reg_strip[(4 * ko_1) * (16) + 8]);
            var0[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[0][1]);
            var0[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[1][1]);
            var0[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[2][1]);
            var0[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[3][1]);
            var0[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[4][1]);
            var0[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            var0[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 4 * ko_1 + 96 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0[0], var1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0[1], var1[1], C_reg[5][1]);
            __m256 var0_1[2];
            __m256 var1_1[2];
            var1_1[0] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16)]);
            var1_1[1] = _mm256_loadu_ps(&B_reg_strip[(1 + 4 * ko_1) * (16) + 8]);
            var0_1[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[0][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[1][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[2][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[3][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[4][1]);
            var0_1[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            var0_1[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 1 + 4 * ko_1 + 96 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_1[0], var1_1[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_1[1], var1_1[1], C_reg[5][1]);
            __m256 var0_2[2];
            __m256 var1_2[2];
            var1_2[0] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16)]);
            var1_2[1] = _mm256_loadu_ps(&B_reg_strip[(2 + 4 * ko_1) * (16) + 8]);
            var0_2[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[0][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[1][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[2][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[3][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[4][1]);
            var0_2[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            var0_2[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 2 + 4 * ko_1 + 96 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_2[0], var1_2[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_2[1], var1_2[1], C_reg[5][1]);
            __m256 var0_3[2];
            __m256 var1_3[2];
            var1_3[0] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16)]);
            var1_3[1] = _mm256_loadu_ps(&B_reg_strip[(3 + 4 * ko_1) * (16) + 8]);
            var0_3[0] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[0][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[0][0]);
            C_reg[0][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[0][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(1 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[1][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[1][0]);
            C_reg[1][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[1][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(2 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[2][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[2][0]);
            C_reg[2][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[2][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(3 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[3][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[3][0]);
            C_reg[3][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[3][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(4 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[4][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[4][0]);
            C_reg[4][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[4][1]);
            var0_3[0] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            var0_3[1] = _mm256_broadcast_ss(&A[(5 + 6 * io_1 + 96 * io) * K + 3 + 4 * ko_1 + 96 * ko]);
            C_reg[5][0] = _mm256_fmadd_ps(var0_3[0], var1_3[0], C_reg[5][0]);
            C_reg[5][1] = _mm256_fmadd_ps(var0_3[1], var1_3[1], C_reg[5][1]);
          }
          _mm256_storeu_ps(&C[(6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[0][0]);
          _mm256_storeu_ps(&C[(6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[0][1]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[1][0]);
          _mm256_storeu_ps(&C[(1 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[1][1]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[2][0]);
          _mm256_storeu_ps(&C[(2 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[2][1]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[3][0]);
          _mm256_storeu_ps(&C[(3 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[3][1]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[4][0]);
          _mm256_storeu_ps(&C[(4 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[4][1]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 96 * io) * N + 16 * jo_1 + 96 * jo], C_reg[5][0]);
          _mm256_storeu_ps(&C[(5 + 6 * io_1 + 96 * io) * N + 8 + 16 * jo_1 + 96 * jo], C_reg[5][1]);
        }
      }
    }
    if (M % 96 > 0) {
      for (int_fast32_t ii = 0; ii < M % 96; ii++) {
        for (int_fast32_t j = 0; j < 96; j++) {
          for (int_fast32_t k = 0; k < 96; k++) {
            C[(ii + (M / 96) * 96) * N + j + 96 * jo] += A[(ii + (M / 96) * 96) * K + k + 96 * ko] * B_strip[k * 96 + j];
          }
        }
      }
    }
  }
}
for (int_fast32_t ko = 0; ko < ((K) / (96)); ko++) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t ji = 0; ji < N % 96; ji++) {
      for (int_fast32_t ki = 0; ki < 96; ki++) {
        C[i * N + ji + (N / 96) * 96] += A[i * K + ki + 96 * ko] * B[(ki + 96 * ko) * N + ji + (N / 96) * 96];
      }
    }
  }
}
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    for (int_fast32_t ki = 0; ki < K % 96; ki++) {
      C[i * N + j] += A[i * K + ki + (K / 96) * 96] * B[(ki + (K / 96) * 96) * N + j];
    }
  }
}
}


/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps(dst,src1,src2)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
