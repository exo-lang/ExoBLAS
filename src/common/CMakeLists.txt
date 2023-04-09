if (NOT HAVE_AVX2)
  set(HAVE_AVX2 0)
endif ()

if (NOT HAVE_NEON)
  set(HAVE_NEON 0)
endif ()

set(SYRK_K_BLK 64 CACHE STRING "K_BLK size for SYRK")
set(SYRK_M_BLK 64 CACHE STRING "M_BLK size for SYRK")
set(SYRK_M_REG 4 CACHE STRING "M_REG size for SYRK")
set(SYRK_N_REG 16 CACHE STRING "N_REG size for SYRK")

set(GEMM_K_BLK 64 CACHE STRING "K_BLK size for GEMM")
set(GEMM_M_BLK 64 CACHE STRING "M_BLK size for GEMM")
set(GEMM_M_REG 4 CACHE STRING "M_REG size for GEMM")
set(GEMM_N_REG 16 CACHE STRING "N_REG size for GEMM")

set(SYMM_K_BLK 64 CACHE STRING "K_BLK size for SYMM")
set(SYMM_M_BLK 64 CACHE STRING "M_BLK size for SYMM")
set(SYMM_M_REG 4 CACHE STRING "M_REG size for SYMM")
set(SYMM_N_REG 16 CACHE STRING "N_REG size for SYMM")


configure_file(exo_blas_config.py.in exo_blas_config.py @ONLY)
