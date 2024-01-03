#include "exo_trmm.h"



#include <stdio.h>
#include <stdlib.h>



// exo_strmm_row_major_Left_Lower_NonTrans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Lower_NonTrans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    float orgBij;
    orgBij = B.data[(m - i - 1) * B.strides[0] + j];
    B.data[(m - i - 1) * B.strides[0] + j] = 0.0;
    for (int_fast32_t k = 0; k < m - i - 1; k++) {
      B.data[(m - i - 1) * B.strides[0] + j] += A.data[(m - i - 1) * A.strides[0] + k] * B.data[k * B.strides[0] + j];
    }
    if (Diag == 0) {
      B.data[(m - i - 1) * B.strides[0] + j] += A.data[(m - i - 1) * A.strides[0] + m - i - 1] * orgBij;
    } else {
      B.data[(m - i - 1) * B.strides[0] + j] += orgBij;
    }
    B.data[(m - i - 1) * B.strides[0] + j] = *alpha * B.data[(m - i - 1) * B.strides[0] + j];
  }
}
}

// exo_strmm_row_major_Left_Lower_Trans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Lower_Trans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    if (Diag == 0) {
      B.data[i * B.strides[0] + j] = A.data[i * A.strides[0] + i] * B.data[i * B.strides[0] + j];
    }
    for (int_fast32_t k = 0; k < m - i - 1; k++) {
      B.data[i * B.strides[0] + j] += A.data[(i + k + 1) * A.strides[0] + i] * B.data[(i + k + 1) * B.strides[0] + j];
    }
    B.data[i * B.strides[0] + j] = *alpha * B.data[i * B.strides[0] + j];
  }
}
}

// exo_strmm_row_major_Left_Upper_NonTrans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Upper_NonTrans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    if (Diag == 0) {
      B.data[i * B.strides[0] + j] = A.data[i * A.strides[0] + i] * B.data[i * B.strides[0] + j];
    }
    for (int_fast32_t k = 0; k < m - i - 1; k++) {
      B.data[i * B.strides[0] + j] += A.data[i * A.strides[0] + i + k + 1] * B.data[(i + k + 1) * B.strides[0] + j];
    }
    B.data[i * B.strides[0] + j] = *alpha * B.data[i * B.strides[0] + j];
  }
}
}

// exo_strmm_row_major_Left_Upper_Trans(
//     m : size,
//     n : size,
//     alpha : f32 @DRAM,
//     A : [f32][m, m] @DRAM,
//     B : [f32][m, n] @DRAM,
//     Diag : size
// )
void exo_strmm_row_major_Left_Upper_Trans( void *ctxt, int_fast32_t m, int_fast32_t n, const float* alpha, struct exo_win_2f32c A, struct exo_win_2f32 B, int_fast32_t Diag ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
for (int_fast32_t j = 0; j < n; j++) {
  for (int_fast32_t i = 0; i < m; i++) {
    float orgBij;
    orgBij = B.data[(m - i - 1) * B.strides[0] + j];
    B.data[(m - i - 1) * B.strides[0] + j] = 0.0;
    for (int_fast32_t k = 0; k < m - i - 1; k++) {
      B.data[(m - i - 1) * B.strides[0] + j] += A.data[k * A.strides[0] + m - i - 1] * B.data[k * B.strides[0] + j];
    }
    if (Diag == 0) {
      B.data[(m - i - 1) * B.strides[0] + j] += A.data[(m - i - 1) * A.strides[0] + m - i - 1] * orgBij;
    } else {
      B.data[(m - i - 1) * B.strides[0] + j] += orgBij;
    }
    B.data[(m - i - 1) * B.strides[0] + j] = *alpha * B.data[(m - i - 1) * B.strides[0] + j];
  }
}
}

