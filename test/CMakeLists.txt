###############################################################################
# Find test dependencies
###############################################################################

##
# BLAS

find_package(BLAS REQUIRED)
target_include_directories(
  BLAS::BLAS INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/common")

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
    EXO_BLAS::exo_${kernel}
    BLAS::BLAS
    benchmark::benchmark_main
  )

  add_test(NAME ${kernel} COMMAND test_${kernel})
  set_tests_properties(
    ${kernel}
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;"
    ENVIRONMENT "BENCHMARK_FORMAT=json;BENCHMARK_OUT=output_${kernel}.json;"
  )
endfunction()

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

