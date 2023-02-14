set(SYRK_K_BLK 64 CACHE STRING "K_BLK size for SYRK")
set(SYRK_M_BLK 64 CACHE STRING "M_BLK size for SYRK")
set(SYRK_M_REG 4 CACHE STRING "M_REG size for SYRK")
set(SYRK_N_REG 16 CACHE STRING "N_REG size for SYRK")

configure_file(exo_blas_config.py.in exo_blas_config.py @ONLY)
