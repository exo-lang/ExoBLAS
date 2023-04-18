#pragma once

#include <cblas.h>
#include <stdio.h>

#include "exo_gemm.h"

void exo_sgemm_notranspose(const int m, const int n, const int k,
                           const float *alpha, const float *beta,
                           const float *A, const float *B, float *C) {
  if (*alpha == 1.0 && *beta == 1.0) {
      //exo_sgemm_notranspose_noalpha_nobeta_main(nullptr, m, n, k, alpha, beta, A, B, C);
       //avx2_microkernel_6x16_1(nullptr, C, A, B);
    if (n <= 48) {
      exo_sgemm_notranspose_noalpha_nobeta_48_48_48(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 96) {
      exo_sgemm_notranspose_noalpha_nobeta_96_96_96(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 192) {
      exo_sgemm_notranspose_noalpha_nobeta_192_192_192(nullptr, m, n, k, alpha,
                                                       beta, A, B, C);
    } else if (n <= 384) {
      exo_sgemm_notranspose_noalpha_nobeta_384_384_384(nullptr, m, n, k, alpha,
                                                       beta, A, B, C);
   // } else if (n <= 768) {
     // exo_sgemm_notranspose_noalpha_nobeta_768_384_384(nullptr, m, n, k, alpha,
      //                                                 beta, A, B, C);
    } else {
      exo_sgemm_notranspose_noalpha_nobeta_768_384_768(nullptr, m, n, k, alpha,
                                                        beta, A, B, C);
    }
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    /*
    if (n <= 32) {
      exo_sgemm_alphazero_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_alphazero_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_alphazero_beta_128_128_128(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 256) {
      exo_sgemm_alphazero_beta_256_256_256(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 512) {
      exo_sgemm_alphazero_beta_512_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 1024) {
      exo_sgemm_alphazero_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 2048) {
      exo_sgemm_alphazero_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 4096) {
      exo_sgemm_alphazero_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else {
      exo_sgemm_alphazero_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    }
    */
  } else if (*alpha != 1.0 && *beta == 1.0) {
    /*
    if (n <= 32) {
      exo_sgemm_notranspose_alpha_nobeta_32_32_32(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 64) {
      exo_sgemm_notranspose_alpha_nobeta_64_64_64(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 128) {
      exo_sgemm_notranspose_alpha_nobeta_128_128_128(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 256) {
      exo_sgemm_notranspose_alpha_nobeta_256_256_256(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 512) {
      exo_sgemm_notranspose_alpha_nobeta_512_256_512(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_notranspose_alpha_nobeta_1024_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_notranspose_alpha_nobeta_2048_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_notranspose_alpha_nobeta_4096_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else {
      exo_sgemm_notranspose_alpha_nobeta_8192_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    }
    */
  } else {
    /*
    if (n <= 32) {
      exo_sgemm_notranspose_alpha_beta_32_32_32(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 64) {
      exo_sgemm_notranspose_alpha_beta_64_64_64(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 128) {
      exo_sgemm_notranspose_alpha_beta_128_128_128(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 256) {
      exo_sgemm_notranspose_alpha_beta_256_256_256(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 512) {
      exo_sgemm_notranspose_alpha_beta_512_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_notranspose_alpha_beta_1024_256_512(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_notranspose_alpha_beta_2048_256_512(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_notranspose_alpha_beta_4096_256_512(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else {
      exo_sgemm_notranspose_alpha_beta_8192_256_512(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    }
    */
  }
}

void exo_sgemm_transa(const int m, const int n, const int k, const float *alpha,
                      const float *beta, const float *A, const float *B,
                      float *C) {
  /*
  if (*alpha == 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transa_noalpha_nobeta_32_32_32(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 64) {
      exo_sgemm_transa_noalpha_nobeta_64_64_64(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 128) {
      exo_sgemm_transa_noalpha_nobeta_128_128_128(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_noalpha_nobeta_256_256_256(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_noalpha_nobeta_512_256_512(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_noalpha_nobeta_1024_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_noalpha_nobeta_2048_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_noalpha_nobeta_4096_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else {
      exo_sgemm_transa_noalpha_nobeta_8192_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    }
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    if (n <= 32) {
      exo_sgemm_alphazero_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_alphazero_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_alphazero_beta_128_128_128(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 256) {
      exo_sgemm_alphazero_beta_256_256_256(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 512) {
      exo_sgemm_alphazero_beta_512_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 1024) {
      exo_sgemm_alphazero_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 2048) {
      exo_sgemm_alphazero_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 4096) {
      exo_sgemm_alphazero_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else {
      exo_sgemm_alphazero_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    }
  } else if (*alpha != 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transa_alpha_nobeta_32_32_32(nullptr, m, n, k, alpha, beta, A,
                                             B, C);
    } else if (n <= 64) {
      exo_sgemm_transa_alpha_nobeta_64_64_64(nullptr, m, n, k, alpha, beta, A,
                                             B, C);
    } else if (n <= 128) {
      exo_sgemm_transa_alpha_nobeta_128_128_128(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_alpha_nobeta_256_256_256(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_alpha_nobeta_512_256_512(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_alpha_nobeta_1024_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_alpha_nobeta_2048_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_alpha_nobeta_4096_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else {
      exo_sgemm_transa_alpha_nobeta_8192_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    }
  } else {
    if (n <= 32) {
      exo_sgemm_transa_alpha_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 64) {
      exo_sgemm_transa_alpha_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 128) {
      exo_sgemm_transa_alpha_beta_128_128_128(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_alpha_beta_256_256_256(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_alpha_beta_512_256_512(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_alpha_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_alpha_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_alpha_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else {
      exo_sgemm_transa_alpha_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    }
  }
*/
}

void exo_sgemm_transb(const int m, const int n, const int k, const float *alpha,
                      const float *beta, const float *A, const float *B,
                      float *C) {
  /*
  if (*alpha == 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transb_noalpha_nobeta_32_32_32(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 64) {
      exo_sgemm_transb_noalpha_nobeta_64_64_64(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 128) {
      exo_sgemm_transb_noalpha_nobeta_128_128_128(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transb_noalpha_nobeta_256_256_256(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transb_noalpha_nobeta_512_256_512(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transb_noalpha_nobeta_1024_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transb_noalpha_nobeta_2048_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transb_noalpha_nobeta_4096_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    } else {
      exo_sgemm_transb_noalpha_nobeta_8192_256_512(nullptr, m, n, k, alpha,
                                                   beta, A, B, C);
    }
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    if (n <= 32) {
      exo_sgemm_alphazero_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_alphazero_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_alphazero_beta_128_128_128(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 256) {
      exo_sgemm_alphazero_beta_256_256_256(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 512) {
      exo_sgemm_alphazero_beta_512_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 1024) {
      exo_sgemm_alphazero_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 2048) {
      exo_sgemm_alphazero_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 4096) {
      exo_sgemm_alphazero_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else {
      exo_sgemm_alphazero_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    }
  } else if (*alpha != 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transb_alpha_nobeta_32_32_32(nullptr, m, n, k, alpha, beta, A,
                                             B, C);
    } else if (n <= 64) {
      exo_sgemm_transb_alpha_nobeta_64_64_64(nullptr, m, n, k, alpha, beta, A,
                                             B, C);
    } else if (n <= 128) {
      exo_sgemm_transb_alpha_nobeta_128_128_128(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transb_alpha_nobeta_256_256_256(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transb_alpha_nobeta_512_256_512(nullptr, m, n, k, alpha, beta,
                                                A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transb_alpha_nobeta_1024_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transb_alpha_nobeta_2048_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transb_alpha_nobeta_4096_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    } else {
      exo_sgemm_transb_alpha_nobeta_8192_256_512(nullptr, m, n, k, alpha, beta,
                                                 A, B, C);
    }
  } else {
    if (n <= 32) {
      exo_sgemm_transb_alpha_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 64) {
      exo_sgemm_transb_alpha_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 128) {
      exo_sgemm_transb_alpha_beta_128_128_128(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 256) {
      exo_sgemm_transb_alpha_beta_256_256_256(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 512) {
      exo_sgemm_transb_alpha_beta_512_256_512(nullptr, m, n, k, alpha, beta, A,
                                              B, C);
    } else if (n <= 1024) {
      exo_sgemm_transb_alpha_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 2048) {
      exo_sgemm_transb_alpha_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else if (n <= 4096) {
      exo_sgemm_transb_alpha_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    } else {
      exo_sgemm_transb_alpha_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A,
                                               B, C);
    }
  }
*/ 
}

void exo_sgemm_transa_transb(const int m, const int n, const int k,
                             const float *alpha, const float *beta,
                             const float *A, const float *B, float *C) {
  /*
  if (*alpha == 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transa_transb_noalpha_nobeta_32_32_32(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_transa_transb_noalpha_nobeta_64_64_64(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_transa_transb_noalpha_nobeta_128_128_128(nullptr, m, n, k,
                                                         alpha, beta, A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_transb_noalpha_nobeta_256_256_256(nullptr, m, n, k,
                                                         alpha, beta, A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_transb_noalpha_nobeta_512_256_512(nullptr, m, n, k,
                                                         alpha, beta, A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_transb_noalpha_nobeta_1024_256_512(nullptr, m, n, k,
                                                          alpha, beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_transb_noalpha_nobeta_2048_256_512(nullptr, m, n, k,
                                                          alpha, beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_transb_noalpha_nobeta_4096_256_512(nullptr, m, n, k,
                                                          alpha, beta, A, B, C);
    } else {
      exo_sgemm_transa_transb_noalpha_nobeta_8192_256_512(nullptr, m, n, k,
                                                          alpha, beta, A, B, C);
    }
  } else if (*alpha == 0.0 && *beta == 1.0) {
    return;
  } else if (*alpha == 0.0 && *beta != 1.0) {
    if (n <= 32) {
      exo_sgemm_alphazero_beta_32_32_32(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_alphazero_beta_64_64_64(nullptr, m, n, k, alpha, beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_alphazero_beta_128_128_128(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 256) {
      exo_sgemm_alphazero_beta_256_256_256(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 512) {
      exo_sgemm_alphazero_beta_512_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                           C);
    } else if (n <= 1024) {
      exo_sgemm_alphazero_beta_1024_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 2048) {
      exo_sgemm_alphazero_beta_2048_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else if (n <= 4096) {
      exo_sgemm_alphazero_beta_4096_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    } else {
      exo_sgemm_alphazero_beta_8192_256_512(nullptr, m, n, k, alpha, beta, A, B,
                                            C);
    }
  } else if (*alpha != 1.0 && *beta == 1.0) {
    if (n <= 32) {
      exo_sgemm_transa_transb_alpha_nobeta_32_32_32(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 64) {
      exo_sgemm_transa_transb_alpha_nobeta_64_64_64(nullptr, m, n, k, alpha,
                                                    beta, A, B, C);
    } else if (n <= 128) {
      exo_sgemm_transa_transb_alpha_nobeta_128_128_128(nullptr, m, n, k, alpha,
                                                       beta, A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_transb_alpha_nobeta_256_256_256(nullptr, m, n, k, alpha,
                                                       beta, A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_transb_alpha_nobeta_512_256_512(nullptr, m, n, k, alpha,
                                                       beta, A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_transb_alpha_nobeta_1024_256_512(nullptr, m, n, k, alpha,
                                                        beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_transb_alpha_nobeta_2048_256_512(nullptr, m, n, k, alpha,
                                                        beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_transb_alpha_nobeta_4096_256_512(nullptr, m, n, k, alpha,
                                                        beta, A, B, C);
    } else {
      exo_sgemm_transa_transb_alpha_nobeta_8192_256_512(nullptr, m, n, k, alpha,
                                                        beta, A, B, C);
    }
  } else {
    if (n <= 32) {
      exo_sgemm_transa_transb_alpha_beta_32_32_32(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 64) {
      exo_sgemm_transa_transb_alpha_beta_64_64_64(nullptr, m, n, k, alpha, beta,
                                                  A, B, C);
    } else if (n <= 128) {
      exo_sgemm_transa_transb_alpha_beta_128_128_128(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 256) {
      exo_sgemm_transa_transb_alpha_beta_256_256_256(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 512) {
      exo_sgemm_transa_transb_alpha_beta_512_256_512(nullptr, m, n, k, alpha,
                                                     beta, A, B, C);
    } else if (n <= 1024) {
      exo_sgemm_transa_transb_alpha_beta_1024_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 2048) {
      exo_sgemm_transa_transb_alpha_beta_2048_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else if (n <= 4096) {
      exo_sgemm_transa_transb_alpha_beta_4096_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    } else {
      exo_sgemm_transa_transb_alpha_beta_8192_256_512(nullptr, m, n, k, alpha,
                                                      beta, A, B, C);
    }
  }
*/
}

void exo_sgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa,
               const enum CBLAS_TRANSPOSE transb, const int m, const int n,
               const int k, const float *alpha, const float *beta,
               const float *A, const float *B, float *C) {
  if (order == CblasColMajor) {
    throw "Unsupported Exception";
  } else {
    if (transa == CblasNoTrans && transb == CblasNoTrans) {
      exo_sgemm_notranspose(m, n, k, alpha, beta, A, B, C);
    } else if (transa == CblasTrans && transb == CblasNoTrans) {
      exo_sgemm_transa(m, n, k, alpha, beta, A, B, C);
    } else if (transa == CblasNoTrans && transb == CblasTrans) {
      exo_sgemm_transb(m, n, k, alpha, beta, A, B, C);
    } else {
      exo_sgemm_transa_transb(m, n, k, alpha, beta, A, B, C);
    }
  }
}
