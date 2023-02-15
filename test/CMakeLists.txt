find_package(BLAS REQUIRED)

find_path(
  CBLAS_INCLUDE_DIR cblas.h REQUIRED
  PATHS /usr/include/x86_64-linux-gnu/openblas-pthread
)
target_include_directories(BLAS::BLAS INTERFACE "${CBLAS_INCLUDE_DIR}")

find_package(benchmark REQUIRED)

if (HAVE_NEON)
  add_executable(test_gemv level2/test_gemv.cpp)
  target_link_libraries(test_gemv PRIVATE EXO_BLAS::gemv BLAS::BLAS benchmark::benchmark_main)
  target_compile_features(test_gemv PRIVATE cxx_std_11)

  add_test(NAME gemv COMMAND test_gemv)

  add_executable(test_syrk level3/test_syrk.cpp)
  target_link_libraries(test_syrk PRIVATE EXO_BLAS::syrk BLAS::BLAS benchmark::benchmark_main)
  target_compile_features(test_syrk PRIVATE cxx_std_11)

  add_test(NAME syrk COMMAND test_syrk )

  add_executable(test_sgemm level3/test_gemm.cpp)
  target_link_libraries(test_sgemm PRIVATE EXO_BLAS::sgemm BLAS::BLAS benchmark::benchmark_main)
  target_compile_features(test_sgemm PRIVATE cxx_std_11)

  add_test(NAME sgemm COMMAND test_sgemm )
endif ()
