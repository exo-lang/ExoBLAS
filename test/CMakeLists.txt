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

## Python
find_package(Python 3 REQUIRED Interpreter)

###############################################################################
# Generic test-case function
###############################################################################

function(add_exo_blas_test kernel precision)
  set(precision_kernel "${precision}${kernel}")

  set(bla_vendor_benchmark_results_dir "${PROJECT_SOURCE_DIR}/benchmark_results/${BLA_VENDOR}")
  file(MAKE_DIRECTORY ${bla_vendor_benchmark_results_dir})
  set(bla_vendor_benchmark_json_output "${bla_vendor_benchmark_results_dir}/${precision_kernel}.json")

  set(exo_benchmark_results_dir ${PROJECT_SOURCE_DIR}/benchmark_results/exo)
  file(MAKE_DIRECTORY ${exo_benchmark_results_dir})
  set(exo_benchmark_json_output "${exo_benchmark_results_dir}/${precision_kernel}.json")

  # Add the benchmark test
  add_executable(${precision_kernel}_bench ${precision_kernel}/bench.cpp)

  target_compile_features(${precision_kernel}_bench PRIVATE cxx_std_11)
  target_link_libraries(
    ${precision_kernel}_bench
    PRIVATE
    EXO_BLAS::exo_${kernel}
    BLAS::BLAS
    benchmark::benchmark_main
  )
  target_include_directories(${precision_kernel}_bench PRIVATE ${precision_kernel})

  add_test(NAME cblas_${precision_kernel}_bench COMMAND ${precision_kernel}_bench --benchmark_filter=cblas|CBLAS)
  set_tests_properties(
    cblas_${precision_kernel}_bench
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;BENCHMARK_FORMAT=console;BENCHMARK_OUT=${bla_vendor_benchmark_json_output}"
  )

  add_test(NAME exo_${precision_kernel}_bench COMMAND ${precision_kernel}_bench --benchmark_filter=exo|EXO)
  set_tests_properties(
    exo_${precision_kernel}_bench
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;BENCHMARK_FORMAT=console;BENCHMARK_OUT=${exo_benchmark_json_output}"
  )

  # Add the correctness test
  add_executable(${precision_kernel}_correctness ${precision_kernel}/correctness.cpp)
  target_compile_features(${precision_kernel}_correctness PRIVATE cxx_std_11)
  target_link_libraries(
    ${precision_kernel}_correctness
    PRIVATE
    EXO_BLAS::exo_${kernel}
    BLAS::BLAS
  )
  target_include_directories(${precision_kernel}_correctness PRIVATE ${precision_kernel})

  add_test(NAME ${precision_kernel}_correctness COMMAND ${precision_kernel}_correctness)
  set_tests_properties(
    ${precision_kernel}_correctness
    PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1"
  )

  add_test(
    NAME ${precision_kernel}_graph
    COMMAND Python::Interpreter "${PROJECT_SOURCE_DIR}/test/graph.py" ${precision_kernel} ${PROJECT_SOURCE_DIR}/benchmark_results ${PROJECT_SOURCE_DIR}/graphs
  )
endfunction()

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
