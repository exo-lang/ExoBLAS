
#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
// syrk_lower_notranspose(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
void syrk_lower_notranspose( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C );




#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>

// gebp_64x128_0(
//     N : size,
//     C : [f32][64, N] @DRAM,
//     A : [f32][64, 128] @DRAM,
//     B : [f32][128, N] @DRAM
// )
static void gebp_64x128_0( void *ctxt, int_fast32_t N, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// gepp_syrk_scheduled(
//     N : size,
//     A1 : [f32][N, 128] @DRAM,
//     A2 : [f32][128, N] @DRAM,
//     C : [f32][N, N] @DRAM
// )
static void gepp_syrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C );

// neon_microkernel_4x16_0(
//     C : [f32][4, 16] @DRAM,
//     A : [f32][4, 128] @DRAM,
//     B : [f32][128, 16] @DRAM
// )
static void neon_microkernel_4x16_0( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B );

// gebp_64x128_0(
//     N : size,
//     C : [f32][64, N] @DRAM,
//     A : [f32][64, 128] @DRAM,
//     B : [f32][128, N] @DRAM
// )
static void gebp_64x128_0( void *ctxt, int_fast32_t N, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
for (int jo = 0; jo < ((N) / (16)); jo++) {
  float *B_strip = malloc(128 * 16 * sizeof(*B_strip));
  for (int i0 = 0; i0 < 128; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      B_strip[(i0) * (16) + (i1) * (1)] = B.data[(i0) * (B.strides[0]) + (i1 + 16 * jo) * (B.strides[1])];
    }
  }
  for (int io = 0; io < 16; io++) {
    neon_microkernel_4x16_0(ctxt,(struct exo_win_2f32){ &C.data[(4 * io) * (C.strides[0]) + (16 * jo) * (C.strides[1])], { C.strides[0], C.strides[1] } },(struct exo_win_2f32c){ &A.data[(4 * io) * (A.strides[0]) + (0) * (A.strides[1])], { A.strides[0], A.strides[1] } },(struct exo_win_2f32c){ &B_strip[(0) * (16) + (0) * (1)], { 16, 1 } });
  }
  free(B_strip);
}
for (int io = 0; io < 16; io++) {
  for (int ii = 0; ii < 4; ii++) {
    if (N % 16 > 0) {
      for (int ji = 0; ji < N % 16; ji++) {
        for (int k = 0; k < 128; k++) {
          C.data[(ii + 4 * io) * (C.strides[0]) + (ji + ((N) / (16)) * 16) * (C.strides[1])] += A.data[(ii + 4 * io) * (A.strides[0]) + (k) * (A.strides[1])] * B.data[(k) * (B.strides[0]) + (ji + ((N) / (16)) * 16) * (B.strides[1])];
        }
      }
    }
  }
}
if (0 > 0) {
  for (int ii = 0; ii < 0; ii++) {
    for (int j = 0; j < 0 + 1 * N; j++) {
      for (int k = 0; k < 128; k++) {
        C.data[(64 + 1 * ii) * (C.strides[0]) + (0 + 1 * j) * (C.strides[1])] += A.data[(64 + 1 * ii) * (A.strides[0]) + (0 + 1 * k) * (A.strides[1])] * B.data[(0 + 1 * k) * (B.strides[0]) + (0 + 1 * j) * (B.strides[1])];
      }
    }
  }
}
}

// gepp_syrk_scheduled(
//     N : size,
//     A1 : [f32][N, 128] @DRAM,
//     A2 : [f32][128, N] @DRAM,
//     C : [f32][N, N] @DRAM
// )
static void gepp_syrk_scheduled( void *ctxt, int_fast32_t N, struct exo_win_2f32c A1, struct exo_win_2f32c A2, struct exo_win_2f32 C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(A1.strides[1] == 1);
EXO_ASSUME(A2.strides[1] == 1);
EXO_ASSUME(C.strides[1] == 1);
for (int io = 0; io < ((N) / (64)); io++) {
  for (int jo = 0; jo < io; jo++) {
    gebp_64x128_0(ctxt,64,(struct exo_win_2f32){ &C.data[(64 * io) * (C.strides[0]) + (1 + 64 * jo) * (C.strides[1])], { C.strides[0], C.strides[1] } },(struct exo_win_2f32c){ &A1.data[(64 * io) * (A1.strides[0]) + (0) * (A1.strides[1])], { A1.strides[0], A1.strides[1] } },(struct exo_win_2f32c){ &A2.data[(0) * (A2.strides[0]) + (1 + 64 * jo) * (A2.strides[1])], { A2.strides[0], A2.strides[1] } });
  }
  for (int ii = 0; ii < 64; ii++) {
    for (int j = 0; j < 1; j++) {
      for (int k = 0; k < 128; k++) {
        C.data[(ii + 64 * io) * (C.strides[0]) + (j) * (C.strides[1])] += A1.data[(ii + 64 * io) * (A1.strides[0]) + (k) * (A1.strides[1])] * A2.data[(k) * (A2.strides[0]) + (j) * (A2.strides[1])];
      }
    }
  }
  for (int ii = 0; ii < 64; ii++) {
    if (ii > 0) {
      for (int ji = 0; ji < ii; ji++) {
        for (int k = 0; k < 128; k++) {
          C.data[(ii + 64 * io) * (C.strides[0]) + (1 + ji + 64 * io) * (C.strides[1])] += A1.data[(ii + 64 * io) * (A1.strides[0]) + (k) * (A1.strides[1])] * A2.data[(k) * (A2.strides[0]) + (1 + ji + 64 * io) * (A2.strides[1])];
        }
      }
    }
  }
}
if (N % 64 > 0) {
  for (int ii = 0; ii < N % 64; ii++) {
    for (int j = 0; j < ii + ((N) / (64)) * 64 + 1; j++) {
      for (int k = 0; k < 128; k++) {
        C.data[(ii + ((N) / (64)) * 64) * (C.strides[0]) + (j) * (C.strides[1])] += A1.data[(ii + ((N) / (64)) * 64) * (A1.strides[0]) + (k) * (A1.strides[1])] * A2.data[(k) * (A2.strides[0]) + (j) * (A2.strides[1])];
      }
    }
  }
}
}


/* relying on the following instruction..."
neon_broadcast_4xf32(dst,src)
{dst_data} = vld1q_dup_f32(&{src_data});
*/
// neon_microkernel_4x16_0(
//     C : [f32][4, 16] @DRAM,
//     A : [f32][4, 128] @DRAM,
//     B : [f32][128, 16] @DRAM
// )
static void neon_microkernel_4x16_0( void *ctxt, struct exo_win_2f32 C, struct exo_win_2f32c A, struct exo_win_2f32c B ) {
float32x4_t C_reg[4][4];
for (int i = 0; i < 4; i++) {
  for (int jo = 0; jo < 4; jo++) {
    C_reg[0][jo + 0] = vld1q_f32(&C.data[(i + 0) * (C.strides[0]) + (4 * jo + 0) * (C.strides[1])]);
  }
}
for (int k = 0; k < 128; k++) {
  float32x4_t A_vec[4];
  for (int i = 0; i < 4; i++) {
    A_vec[0] = vld1q_dup_f32(&A.data[(i + 0) * (A.strides[0]) + (k + 0) * (A.strides[1])]);
  }
  float32x4_t B_vec[4];
  for (int jo = 0; jo < 4; jo++) {
    B_vec[0] = vld1q_f32(&B.data[(k + 0) * (B.strides[0]) + (4 * jo + 0) * (B.strides[1])]);
  }
  for (int i = 0; i < 4; i++) {
    for (int jo = 0; jo < 4; jo++) {
      C_reg[0][jo + 0] = vmlaq_f32(C_reg[0][jo + 0], A_vec[0], B_vec[0]);
    }
  }
}
for (int i = 0; i < 4; i++) {
  for (int jo = 0; jo < 4; jo++) {
    vst1q_f32(&C.data[(i + 0) * (C.strides[0]) + (4 * jo + 0) * (C.strides[1])], C_reg[0][jo + 0]);
  }
}
}


/* relying on the following instruction..."
neon_vfmadd_4xf32_4xf32(dst,lhs,rhs)
{dst_data} = vmlaq_f32({dst_data}, {lhs_data}, {rhs_data});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/
// syrk_lower_notranspose(
//     N : size,
//     K : size,
//     A1 : f32[N, K] @DRAM,
//     A2 : f32[K, N] @DRAM,
//     C : f32[N, N] @DRAM
// )
void syrk_lower_notranspose( void *ctxt, int_fast32_t N, int_fast32_t K, const float* A1, const float* A2, float* C ) {
EXO_ASSUME(N >= 1);
EXO_ASSUME(K >= 1);
EXO_ASSUME(1 == 1);
EXO_ASSUME(1 == 1);
EXO_ASSUME(1 == 1);
for (int ko = 0; ko < ((K) / (128)); ko++) {
  gepp_syrk_scheduled(ctxt,N + 0,(struct exo_win_2f32c){ &A1[(0) * (K) + (128 * ko + 0) * (1)], { K, 1 } },(struct exo_win_2f32c){ &A2[(128 * ko + 0) * (N) + (0) * (1)], { N, 1 } },(struct exo_win_2f32){ &C[(0) * (N) + (0) * (1)], { N, 1 } });
}
for (int i = 0; i < N; i++) {
  for (int j = 0; j < i + 1; j++) {
    if (K % 128 > 0) {
      for (int ki = 0; ki < K % 128; ki++) {
        C[(i) * (N) + (j) * (1)] += A1[(i) * (K) + (ki + ((K) / (128)) * 128) * (1)] * A2[(ki + ((K) / (128)) * 128) * (N) + (j) * (1)];
      }
    }
  }
}
}

