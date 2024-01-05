#include "exo_gemm.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>


// exo_sgemm_matmul_stride_any_no_mem_sys_tiling(
//     M : size,
//     N : size,
//     K : size,
//     A : [f32][M, K] @DRAM,
//     B : [f32][K, N] @DRAM,
//     C : [f32][M, N] @DRAM
// )
static void exo_sgemm_matmul_stride_any_no_mem_sys_tiling( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C );


/* relying on the following instruction..."
avx2_reduce_add_wide_ps(dst,src)
{dst_data} = _mm256_add_ps({src_data}, {dst_data});
*/

/* relying on the following instruction..."
avx2_reg_copy_ps(dst,src)
{dst_data} = {src_data};
*/
// exo_sgemm_matmul_stride_any(
//     M : size,
//     N : size,
//     K : size,
//     A : [f32][M, K] @DRAM,
//     B : [f32][K, N] @DRAM,
//     C : [f32][M, N] @DRAM
// )
void exo_sgemm_matmul_stride_any( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
for (int_fast32_t ko = 0; ko < ((K) / (192)); ko++) {
  for (int_fast32_t io = 0; io < ((M) / (4096)); io++) {
    for (int_fast32_t jo = 0; jo < ((N) / (3072)); jo++) {
      exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,4096,3072,192,(struct exo_win_2f32c){ &A.data[(4096 * io) * (A.strides[0]) + 192 * ko], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * ko) * (B.strides[0]) + 3072 * jo], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * io) * (C.strides[0]) + 3072 * jo], { C.strides[0], 1 } });
    }
    exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,4096,N % 3072,192,(struct exo_win_2f32c){ &A.data[(4096 * io) * (A.strides[0]) + 192 * ko], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * ko) * (B.strides[0]) + 3072 * (N / 3072)], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * io) * (C.strides[0]) + 3072 * (N / 3072)], { C.strides[0], 1 } });
  }
  for (int_fast32_t jo = 0; jo < ((N) / (3072)); jo++) {
    exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,M % 4096,3072,192,(struct exo_win_2f32c){ &A.data[(4096 * (M / 4096)) * (A.strides[0]) + 192 * ko], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * ko) * (B.strides[0]) + 3072 * jo], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * (M / 4096)) * (C.strides[0]) + 3072 * jo], { C.strides[0], 1 } });
  }
  exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,M % 4096,N % 3072,192,(struct exo_win_2f32c){ &A.data[(4096 * (M / 4096)) * (A.strides[0]) + 192 * ko], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * ko) * (B.strides[0]) + 3072 * (N / 3072)], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * (M / 4096)) * (C.strides[0]) + 3072 * (N / 3072)], { C.strides[0], 1 } });
}
for (int_fast32_t io = 0; io < ((M) / (4096)); io++) {
  for (int_fast32_t jo = 0; jo < ((N) / (3072)); jo++) {
    exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,4096,3072,K % 192,(struct exo_win_2f32c){ &A.data[(4096 * io) * (A.strides[0]) + 192 * (K / 192)], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * (K / 192)) * (B.strides[0]) + 3072 * jo], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * io) * (C.strides[0]) + 3072 * jo], { C.strides[0], 1 } });
  }
  exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,4096,N % 3072,K % 192,(struct exo_win_2f32c){ &A.data[(4096 * io) * (A.strides[0]) + 192 * (K / 192)], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * (K / 192)) * (B.strides[0]) + 3072 * (N / 3072)], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * io) * (C.strides[0]) + 3072 * (N / 3072)], { C.strides[0], 1 } });
}
for (int_fast32_t jo = 0; jo < ((N) / (3072)); jo++) {
  exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,M % 4096,3072,K % 192,(struct exo_win_2f32c){ &A.data[(4096 * (M / 4096)) * (A.strides[0]) + 192 * (K / 192)], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * (K / 192)) * (B.strides[0]) + 3072 * jo], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * (M / 4096)) * (C.strides[0]) + 3072 * jo], { C.strides[0], 1 } });
}
exo_sgemm_matmul_stride_any_no_mem_sys_tiling(ctxt,M % 4096,N % 3072,K % 192,(struct exo_win_2f32c){ &A.data[(4096 * (M / 4096)) * (A.strides[0]) + 192 * (K / 192)], { A.strides[0], 1 } },(struct exo_win_2f32c){ &B.data[(192 * (K / 192)) * (B.strides[0]) + 3072 * (N / 3072)], { B.strides[0], 1 } },(struct exo_win_2f32){ &C.data[(4096 * (M / 4096)) * (C.strides[0]) + 3072 * (N / 3072)], { C.strides[0], 1 } });
}

// exo_sgemm_matmul_stride_any_no_mem_sys_tiling(
//     M : size,
//     N : size,
//     K : size,
//     A : [f32][M, K] @DRAM,
//     B : [f32][K, N] @DRAM,
//     C : [f32][M, N] @DRAM
// )
static void exo_sgemm_matmul_stride_any_no_mem_sys_tiling( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, struct exo_win_2f32c A, struct exo_win_2f32c B, struct exo_win_2f32 C ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
EXO_ASSUME(K <= 192);
EXO_ASSUME(M <= 4096);
EXO_ASSUME(N <= 3072);
static float B_repacked_access_order[128 * 192 * 3 * 8];
if (((M) / (4)) > 0) {
  for (int_fast32_t jo = 0; jo < ((N) / (24)); jo++) {
    for (int_fast32_t k = 0; k < K; k++) {
      __m256 reg[3];
      reg[0] = _mm256_loadu_ps(&B.data[(k) * (B.strides[0]) + 24 * jo]);
      reg[1] = _mm256_loadu_ps(&B.data[(k) * (B.strides[0]) + 8 + 24 * jo]);
      reg[2] = _mm256_loadu_ps(&B.data[(k) * (B.strides[0]) + 16 + 24 * jo]);
      _mm256_storeu_ps(&B_repacked_access_order[(jo) * (4608) + (k) * (24)], reg[0]);
      _mm256_storeu_ps(&B_repacked_access_order[(jo) * (4608) + (k) * (24) + 8], reg[1]);
      _mm256_storeu_ps(&B_repacked_access_order[(jo) * (4608) + (k) * (24) + (2) * 8], reg[2]);
    }
  }
}
static float A_repacked_access_order[1024 * 192 * 4];
if (((N) / (24)) > 0) {
  for (int_fast32_t io = 0; io < ((M) / (4)); io++) {
    for (int_fast32_t k = 0; k < K; k++) {
      A_repacked_access_order[io * 768 + k * 4] = A.data[4 * io * A.strides[0] + k];
      A_repacked_access_order[io * 768 + k * 4 + 1] = A.data[(1 + 4 * io) * A.strides[0] + k];
      A_repacked_access_order[io * 768 + k * 4 + 2] = A.data[(2 + 4 * io) * A.strides[0] + k];
      A_repacked_access_order[io * 768 + k * 4 + 3] = A.data[(3 + 4 * io) * A.strides[0] + k];
    }
  }
}
for (int_fast32_t io = 0; io < ((M) / (4)); io++) {
  for (int_fast32_t jo = 0; jo < ((N) / (24)); jo++) {
    __m256 C_reg[4][3];
    __m256 reg;
    reg = _mm256_setzero_ps();
    C_reg[0][0] = reg;
    __m256 reg_1;
    reg_1 = _mm256_setzero_ps();
    C_reg[0][1] = reg_1;
    __m256 reg_2;
    reg_2 = _mm256_setzero_ps();
    C_reg[0][2] = reg_2;
    __m256 reg_3;
    reg_3 = _mm256_setzero_ps();
    C_reg[1][0] = reg_3;
    __m256 reg_4;
    reg_4 = _mm256_setzero_ps();
    C_reg[1][1] = reg_4;
    __m256 reg_5;
    reg_5 = _mm256_setzero_ps();
    C_reg[1][2] = reg_5;
    __m256 reg_6;
    reg_6 = _mm256_setzero_ps();
    C_reg[2][0] = reg_6;
    __m256 reg_7;
    reg_7 = _mm256_setzero_ps();
    C_reg[2][1] = reg_7;
    __m256 reg_8;
    reg_8 = _mm256_setzero_ps();
    C_reg[2][2] = reg_8;
    __m256 reg_9;
    reg_9 = _mm256_setzero_ps();
    C_reg[3][0] = reg_9;
    __m256 reg_10;
    reg_10 = _mm256_setzero_ps();
    C_reg[3][1] = reg_10;
    __m256 reg_11;
    reg_11 = _mm256_setzero_ps();
    C_reg[3][2] = reg_11;
    for (int_fast32_t ko = 0; ko < ((K) / (4)); ko++) {
      __m256 B_reg[3];
      B_reg[0] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (4 * ko) * (24)]);
      B_reg[1] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (4 * ko) * (24) + 8]);
      B_reg[2] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (4 * ko) * (24) + (2) * 8]);
      __m256 reg_12;
      reg_12 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (4 * ko) * 4]);
      C_reg[0][0] = _mm256_fmadd_ps(reg_12, B_reg[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(reg_12, B_reg[1], C_reg[0][1]);
      C_reg[0][2] = _mm256_fmadd_ps(reg_12, B_reg[2], C_reg[0][2]);
      __m256 reg_13;
      reg_13 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (4 * ko) * 4 + 1]);
      C_reg[1][0] = _mm256_fmadd_ps(reg_13, B_reg[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(reg_13, B_reg[1], C_reg[1][1]);
      C_reg[1][2] = _mm256_fmadd_ps(reg_13, B_reg[2], C_reg[1][2]);
      __m256 reg_14;
      reg_14 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (4 * ko) * 4 + 2]);
      C_reg[2][0] = _mm256_fmadd_ps(reg_14, B_reg[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(reg_14, B_reg[1], C_reg[2][1]);
      C_reg[2][2] = _mm256_fmadd_ps(reg_14, B_reg[2], C_reg[2][2]);
      __m256 reg_15;
      reg_15 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (4 * ko) * 4 + 3]);
      C_reg[3][0] = _mm256_fmadd_ps(reg_15, B_reg[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(reg_15, B_reg[1], C_reg[3][1]);
      C_reg[3][2] = _mm256_fmadd_ps(reg_15, B_reg[2], C_reg[3][2]);
      __m256 B_reg_1[3];
      B_reg_1[0] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (1 + 4 * ko) * (24)]);
      B_reg_1[1] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (1 + 4 * ko) * (24) + 8]);
      B_reg_1[2] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (1 + 4 * ko) * (24) + (2) * 8]);
      __m256 reg_16;
      reg_16 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (1 + 4 * ko) * 4]);
      C_reg[0][0] = _mm256_fmadd_ps(reg_16, B_reg_1[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(reg_16, B_reg_1[1], C_reg[0][1]);
      C_reg[0][2] = _mm256_fmadd_ps(reg_16, B_reg_1[2], C_reg[0][2]);
      __m256 reg_17;
      reg_17 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (1 + 4 * ko) * 4 + 1]);
      C_reg[1][0] = _mm256_fmadd_ps(reg_17, B_reg_1[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(reg_17, B_reg_1[1], C_reg[1][1]);
      C_reg[1][2] = _mm256_fmadd_ps(reg_17, B_reg_1[2], C_reg[1][2]);
      __m256 reg_18;
      reg_18 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (1 + 4 * ko) * 4 + 2]);
      C_reg[2][0] = _mm256_fmadd_ps(reg_18, B_reg_1[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(reg_18, B_reg_1[1], C_reg[2][1]);
      C_reg[2][2] = _mm256_fmadd_ps(reg_18, B_reg_1[2], C_reg[2][2]);
      __m256 reg_19;
      reg_19 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (1 + 4 * ko) * 4 + 3]);
      C_reg[3][0] = _mm256_fmadd_ps(reg_19, B_reg_1[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(reg_19, B_reg_1[1], C_reg[3][1]);
      C_reg[3][2] = _mm256_fmadd_ps(reg_19, B_reg_1[2], C_reg[3][2]);
      __m256 B_reg_2[3];
      B_reg_2[0] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (2 + 4 * ko) * (24)]);
      B_reg_2[1] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (2 + 4 * ko) * (24) + 8]);
      B_reg_2[2] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (2 + 4 * ko) * (24) + (2) * 8]);
      __m256 reg_20;
      reg_20 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (2 + 4 * ko) * 4]);
      C_reg[0][0] = _mm256_fmadd_ps(reg_20, B_reg_2[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(reg_20, B_reg_2[1], C_reg[0][1]);
      C_reg[0][2] = _mm256_fmadd_ps(reg_20, B_reg_2[2], C_reg[0][2]);
      __m256 reg_21;
      reg_21 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (2 + 4 * ko) * 4 + 1]);
      C_reg[1][0] = _mm256_fmadd_ps(reg_21, B_reg_2[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(reg_21, B_reg_2[1], C_reg[1][1]);
      C_reg[1][2] = _mm256_fmadd_ps(reg_21, B_reg_2[2], C_reg[1][2]);
      __m256 reg_22;
      reg_22 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (2 + 4 * ko) * 4 + 2]);
      C_reg[2][0] = _mm256_fmadd_ps(reg_22, B_reg_2[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(reg_22, B_reg_2[1], C_reg[2][1]);
      C_reg[2][2] = _mm256_fmadd_ps(reg_22, B_reg_2[2], C_reg[2][2]);
      __m256 reg_23;
      reg_23 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (2 + 4 * ko) * 4 + 3]);
      C_reg[3][0] = _mm256_fmadd_ps(reg_23, B_reg_2[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(reg_23, B_reg_2[1], C_reg[3][1]);
      C_reg[3][2] = _mm256_fmadd_ps(reg_23, B_reg_2[2], C_reg[3][2]);
      __m256 B_reg_3[3];
      B_reg_3[0] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (3 + 4 * ko) * (24)]);
      B_reg_3[1] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (3 + 4 * ko) * (24) + 8]);
      B_reg_3[2] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (3 + 4 * ko) * (24) + (2) * 8]);
      __m256 reg_24;
      reg_24 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (3 + 4 * ko) * 4]);
      C_reg[0][0] = _mm256_fmadd_ps(reg_24, B_reg_3[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(reg_24, B_reg_3[1], C_reg[0][1]);
      C_reg[0][2] = _mm256_fmadd_ps(reg_24, B_reg_3[2], C_reg[0][2]);
      __m256 reg_25;
      reg_25 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (3 + 4 * ko) * 4 + 1]);
      C_reg[1][0] = _mm256_fmadd_ps(reg_25, B_reg_3[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(reg_25, B_reg_3[1], C_reg[1][1]);
      C_reg[1][2] = _mm256_fmadd_ps(reg_25, B_reg_3[2], C_reg[1][2]);
      __m256 reg_26;
      reg_26 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (3 + 4 * ko) * 4 + 2]);
      C_reg[2][0] = _mm256_fmadd_ps(reg_26, B_reg_3[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(reg_26, B_reg_3[1], C_reg[2][1]);
      C_reg[2][2] = _mm256_fmadd_ps(reg_26, B_reg_3[2], C_reg[2][2]);
      __m256 reg_27;
      reg_27 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (3 + 4 * ko) * 4 + 3]);
      C_reg[3][0] = _mm256_fmadd_ps(reg_27, B_reg_3[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(reg_27, B_reg_3[1], C_reg[3][1]);
      C_reg[3][2] = _mm256_fmadd_ps(reg_27, B_reg_3[2], C_reg[3][2]);
    }
    for (int_fast32_t ki = 0; ki < K % 4; ki++) {
      __m256 B_reg[3];
      B_reg[0] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (ki + (K / 4) * 4) * (24)]);
      B_reg[1] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (ki + (K / 4) * 4) * (24) + 8]);
      B_reg[2] = _mm256_loadu_ps(&B_repacked_access_order[(jo) * (4608) + (ki + (K / 4) * 4) * (24) + (2) * 8]);
      __m256 reg_12;
      reg_12 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (ki + (K / 4) * 4) * 4]);
      C_reg[0][0] = _mm256_fmadd_ps(reg_12, B_reg[0], C_reg[0][0]);
      C_reg[0][1] = _mm256_fmadd_ps(reg_12, B_reg[1], C_reg[0][1]);
      C_reg[0][2] = _mm256_fmadd_ps(reg_12, B_reg[2], C_reg[0][2]);
      __m256 reg_13;
      reg_13 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (ki + (K / 4) * 4) * 4 + 1]);
      C_reg[1][0] = _mm256_fmadd_ps(reg_13, B_reg[0], C_reg[1][0]);
      C_reg[1][1] = _mm256_fmadd_ps(reg_13, B_reg[1], C_reg[1][1]);
      C_reg[1][2] = _mm256_fmadd_ps(reg_13, B_reg[2], C_reg[1][2]);
      __m256 reg_14;
      reg_14 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (ki + (K / 4) * 4) * 4 + 2]);
      C_reg[2][0] = _mm256_fmadd_ps(reg_14, B_reg[0], C_reg[2][0]);
      C_reg[2][1] = _mm256_fmadd_ps(reg_14, B_reg[1], C_reg[2][1]);
      C_reg[2][2] = _mm256_fmadd_ps(reg_14, B_reg[2], C_reg[2][2]);
      __m256 reg_15;
      reg_15 = _mm256_broadcast_ss(&A_repacked_access_order[(io) * (768) + (ki + (K / 4) * 4) * 4 + 3]);
      C_reg[3][0] = _mm256_fmadd_ps(reg_15, B_reg[0], C_reg[3][0]);
      C_reg[3][1] = _mm256_fmadd_ps(reg_15, B_reg[1], C_reg[3][1]);
      C_reg[3][2] = _mm256_fmadd_ps(reg_15, B_reg[2], C_reg[3][2]);
    }
    __m256 reg_12[3][4];
    reg_12[0][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 24 * jo]);
    reg_12[1][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 24 * jo]);
    reg_12[2][0] = _mm256_loadu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 + 24 * jo]);
    reg_12[0][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 24 * jo]);
    reg_12[1][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 24 * jo]);
    reg_12[2][1] = _mm256_loadu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 + 24 * jo]);
    reg_12[0][2] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 24 * jo]);
    reg_12[1][2] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 24 * jo]);
    reg_12[2][2] = _mm256_loadu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 + 24 * jo]);
    reg_12[0][3] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 24 * jo]);
    reg_12[1][3] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 24 * jo]);
    reg_12[2][3] = _mm256_loadu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 + 24 * jo]);
    reg_12[0][0] = _mm256_add_ps(C_reg[0][0], reg_12[0][0]);
    reg_12[1][0] = _mm256_add_ps(C_reg[0][1], reg_12[1][0]);
    reg_12[2][0] = _mm256_add_ps(C_reg[0][2], reg_12[2][0]);
    reg_12[0][1] = _mm256_add_ps(C_reg[1][0], reg_12[0][1]);
    reg_12[1][1] = _mm256_add_ps(C_reg[1][1], reg_12[1][1]);
    reg_12[2][1] = _mm256_add_ps(C_reg[1][2], reg_12[2][1]);
    reg_12[0][2] = _mm256_add_ps(C_reg[2][0], reg_12[0][2]);
    reg_12[1][2] = _mm256_add_ps(C_reg[2][1], reg_12[1][2]);
    reg_12[2][2] = _mm256_add_ps(C_reg[2][2], reg_12[2][2]);
    reg_12[0][3] = _mm256_add_ps(C_reg[3][0], reg_12[0][3]);
    reg_12[1][3] = _mm256_add_ps(C_reg[3][1], reg_12[1][3]);
    reg_12[2][3] = _mm256_add_ps(C_reg[3][2], reg_12[2][3]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 24 * jo], reg_12[0][0]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 8 + 24 * jo], reg_12[1][0]);
    _mm256_storeu_ps(&C.data[(4 * io) * (C.strides[0]) + 16 + 24 * jo], reg_12[2][0]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 24 * jo], reg_12[0][1]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 8 + 24 * jo], reg_12[1][1]);
    _mm256_storeu_ps(&C.data[(1 + 4 * io) * (C.strides[0]) + 16 + 24 * jo], reg_12[2][1]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 24 * jo], reg_12[0][2]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 8 + 24 * jo], reg_12[1][2]);
    _mm256_storeu_ps(&C.data[(2 + 4 * io) * (C.strides[0]) + 16 + 24 * jo], reg_12[2][2]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 24 * jo], reg_12[0][3]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 8 + 24 * jo], reg_12[1][3]);
    _mm256_storeu_ps(&C.data[(3 + 4 * io) * (C.strides[0]) + 16 + 24 * jo], reg_12[2][3]);
  }
}
for (int_fast32_t k = 0; k < K; k++) {
  for (int_fast32_t i = 0; i < ((M) / (4)) * 4; i++) {
    for (int_fast32_t ji = 0; ji < N % 24; ji++) {
      C.data[i * C.strides[0] + ji + (N / 24) * 24] += A.data[i * A.strides[0] + k] * B.data[k * B.strides[0] + ji + (N / 24) * 24];
    }
  }
}
for (int_fast32_t k = 0; k < K; k++) {
  for (int_fast32_t ii = 0; ii < M % 4; ii++) {
    for (int_fast32_t j = 0; j < N; j++) {
      C.data[(ii + (M / 4) * 4) * C.strides[0] + j] += A.data[(ii + (M / 4) * 4) * A.strides[0] + k] * B.data[k * B.strides[0] + j];
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
mm256_setzero_ps(dst)
{dst_data} = _mm256_setzero_ps();
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
