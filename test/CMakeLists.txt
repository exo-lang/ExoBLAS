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
  # Add the benchmark test
  add_executable(${kernel}_bench ${kernel}/bench.cpp)

  target_compile_features(${kernel}_bench PRIVATE cxx_std_11)
  target_link_libraries(
    ${kernel}_bench
    PRIVATE
    EXO_BLAS::exo_${kernel}
    BLAS::BLAS
    benchmark::benchmark_main
  )

  add_test(NAME ${kernel}_bench COMMAND ${kernel}_bench)
  set_tests_properties(
    ${kernel}_bench
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;"
  )

  # Add the correctness test
  add_executable(${kernel}_correctness ${kernel}/correctness.cpp)
  target_compile_features(${kernel}_correctness PRIVATE cxx_std_11)
  target_link_libraries(
    ${kernel}_correctness
    PRIVATE
    EXO_BLAS::exo_${kernel}
    BLAS::BLAS
  )

  add_test(NAME ${kernel}_correctness COMMAND ${kernel}_correctness)
  set_tests_properties(
    ${kernel}_bench
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;"
    ENVIRONMENT "BENCHMARK_FORMAT=json;BENCHMARK_OUT=output_${kernel}.json;"
  )
endfunction()

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

