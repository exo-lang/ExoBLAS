#include "sgemv.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>



/* relying on the following instruction..."
neon_broadcast_4xf32(dst,src)
{dst_data} = vld1q_dup_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vfmla_4xf32_4xf32(dst,lhs,rhs,lane)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vmul_4xf32(dst,lhs,rhs)
{dst_data} = vmulq_f32({lhs_data}, {rhs_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/
// sgemv_exo(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     m : size,
//     n : size,
//     lda : size,
//     a : f32[m, lda] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM
// )
void sgemv_exo( void *ctxt, const float* alpha, const float* beta, int_fast32_t m, int_fast32_t n, int_fast32_t lda, const float* a, const float* x, float* y ) {
EXO_ASSUME(n <= lda);
float32x4_t beta_vec;
float *beta_temp = malloc(1 * sizeof(*beta_temp));
beta_temp[(0) * (1)] = *beta;
beta_vec = vld1q_dup_f32(&beta_temp[(0) * (1)]);
free(beta_temp);
for (int io = 0; io < ((m) / (4)); io++) {
  float32x4_t old_y_vec;
  old_y_vec = vld1q_f32(&y[(4 * io) * (1)]);
  float32x4_t new_y_vec;
  new_y_vec = vmulq_f32(beta_vec, old_y_vec);
  vst1q_f32(&y[(4 * io) * (1)], new_y_vec);
}
float32x4_t alpha_vec;
float *alpha_temp = malloc(1 * sizeof(*alpha_temp));
alpha_temp[(0) * (1)] = *alpha;
alpha_vec = vld1q_dup_f32(&alpha_temp[(0) * (1)]);
free(alpha_temp);
float *a_transposed = malloc(4 * 4 * sizeof(*a_transposed));
for (int io = 0; io < ((m) / (4)); io++) {
  float32x4_t y_vec;
  y_vec = vld1q_f32(&y[(4 * io) * (1)]);
  for (int jo = 0; jo < ((n) / (4)); jo++) {
    float32x4_t alpha_times_x;
    float32x4_t x_vec;
    x_vec = vld1q_f32(&x[(4 * jo) * (1)]);
    alpha_times_x = vmulq_f32(alpha_vec, x_vec);
    float32x4_t a_vecs[4];
    for (int i0 = 0; i0 < 4; i0++) {
      for (int i1 = 0; i1 < 4; i1++) {
        a_transposed[(i1) * (4) + (i0) * (1)] = a[(i0 + 4 * io) * (lda) + (i1 + 4 * jo) * (1)];
      }
    }
    for (int i1 = 0; i1 < 4; i1++) {
      a_vecs[i1] = vld1q_f32(&a_transposed[(i1) * (4) + (0) * (1)]);
    }
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[0], alpha_times_x, (0));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[1], alpha_times_x, (1));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[2], alpha_times_x, (2));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[3], alpha_times_x, (3));
  }
  vst1q_f32(&y[(4 * io) * (1)], y_vec);
  for (int ii = 0; ii < 4; ii++) {
    if (n % 4 > 0) {
      for (int ji = 0; ji < n % 4; ji++) {
        y[(ii + 4 * io) * (1)] += *alpha * x[(ji + ((n) / (4)) * 4) * (1)] * a[(ii + 4 * io) * (lda) + (ji + ((n) / (4)) * 4) * (1)];
      }
    }
  }
}
free(a_transposed);
if (m % 4 > 0) {
  for (int ii = 0; ii < m % 4; ii++) {
    y[(ii + ((m) / (4)) * 4) * (1)] = *beta * y[(ii + ((m) / (4)) * 4) * (1)];
    for (int j = 0; j < n; j++) {
      y[(ii + ((m) / (4)) * 4) * (1)] += *alpha * x[(j) * (1)] * a[(ii + ((m) / (4)) * 4) * (lda) + (j) * (1)];
    }
  }
}
}

// sgemv_transpose_exo(
//     alpha : f32 @DRAM,
//     beta : f32 @DRAM,
//     n : size,
//     m : size,
//     lda : size,
//     a : f32[n, lda] @DRAM,
//     x : f32[n] @DRAM,
//     y : f32[m] @DRAM
// )
void sgemv_transpose_exo( void *ctxt, const float* alpha, const float* beta, int_fast32_t n, int_fast32_t m, int_fast32_t lda, const float* a, const float* x, float* y ) {
EXO_ASSUME(m <= lda);
float32x4_t beta_vec;
float *beta_temp = malloc(1 * sizeof(*beta_temp));
beta_temp[(0) * (1)] = *beta;
beta_vec = vld1q_dup_f32(&beta_temp[(0) * (1)]);
free(beta_temp);
for (int io = 0; io < ((m) / (4)); io++) {
  float32x4_t old_y_vec;
  old_y_vec = vld1q_f32(&y[(4 * io) * (1)]);
  float32x4_t new_y_vec;
  new_y_vec = vmulq_f32(beta_vec, old_y_vec);
  vst1q_f32(&y[(4 * io) * (1)], new_y_vec);
}
float32x4_t alpha_vec;
float *alpha_temp = malloc(1 * sizeof(*alpha_temp));
alpha_temp[(0) * (1)] = *alpha;
alpha_vec = vld1q_dup_f32(&alpha_temp[(0) * (1)]);
free(alpha_temp);
for (int io = 0; io < ((m) / (4)); io++) {
  float32x4_t y_vec;
  y_vec = vld1q_f32(&y[(4 * io) * (1)]);
  for (int jo = 0; jo < ((n) / (4)); jo++) {
    float32x4_t alpha_times_x;
    float32x4_t x_vec;
    x_vec = vld1q_f32(&x[(4 * jo) * (1)]);
    alpha_times_x = vmulq_f32(alpha_vec, x_vec);
    float32x4_t a_vecs[4];
    for (int i0 = 0; i0 < 4; i0++) {
      a_vecs[i0] = vld1q_f32(&a[(i0 + 4 * jo) * (lda) + (4 * io) * (1)]);
    }
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[0], alpha_times_x, (0));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[1], alpha_times_x, (1));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[2], alpha_times_x, (2));
    y_vec = vfmaq_laneq_f32(y_vec, a_vecs[3], alpha_times_x, (3));
  }
  vst1q_f32(&y[(4 * io) * (1)], y_vec);
  for (int ii = 0; ii < 4; ii++) {
    if (n % 4 > 0) {
      for (int ji = 0; ji < n % 4; ji++) {
        y[(ii + 4 * io) * (1)] += *alpha * x[(ji + ((n) / (4)) * 4) * (1)] * a[(ji + ((n) / (4)) * 4) * (lda) + (ii + 4 * io) * (1)];
      }
    }
  }
}
if (m % 4 > 0) {
  for (int ii = 0; ii < m % 4; ii++) {
    y[(ii + ((m) / (4)) * 4) * (1)] = *beta * y[(ii + ((m) / (4)) * 4) * (1)];
    for (int j = 0; j < n; j++) {
      y[(ii + ((m) / (4)) * 4) * (1)] += *alpha * x[(j) * (1)] * a[(j) * (lda) + (ii + ((m) / (4)) * 4) * (1)];
    }
  }
}
}

