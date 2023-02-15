###############################################################################
# Find test dependencies
###############################################################################

##
# BLAS

find_package(BLAS REQUIRED)

# Attach cblas header to BLAS::BLAS target
find_path(
  CBLAS_INCLUDE_DIR cblas.h REQUIRED
  PATHS /usr/include/x86_64-linux-gnu/openblas-pthread
)
target_include_directories(BLAS::BLAS INTERFACE "${CBLAS_INCLUDE_DIR}")

##
# Google Benchmark

find_package(benchmark REQUIRED)

###############################################################################
# Generic test-case function
###############################################################################

function(add_exo_blas_test kernel)
  add_executable(test_${kernel} test_${kernel}.cpp)
  target_compile_features(test_${kernel} PRIVATE cxx_std_11)
  target_link_libraries(
    test_${kernel}
    PRIVATE
    EXO_BLAS::${kernel}
    BLAS::BLAS
    benchmark::benchmark_main
  )

  add_test(NAME ${kernel} COMMAND test_${kernel})
  set_tests_properties(
    ${kernel}
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
  )
endfunction()

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

