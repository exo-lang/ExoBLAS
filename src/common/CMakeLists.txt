if (NOT HAVE_AVX2)
  set(HAVE_AVX2 0)
endif ()

if (NOT HAVE_NEON)
  set(HAVE_NEON 0)
endif ()

configure_file(exo_blas_config.py.in exo_blas_config.py @ONLY)
