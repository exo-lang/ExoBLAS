cmake_minimum_required(VERSION 3.25)
project(EXO_BLAS)

enable_testing()

include(CheckSourceCompiles)

check_source_compiles(C [[
  #include <immintrin.h>
  int main() {
    __m256 v = _mm256_set1_ps(1.0);
    _mm256_fmadd_ps(v, v, v);
  }
  ]] HAVE_AVX2)
check_source_compiles(C [[
  #include <arm_neon.h>
  int main() {
    float32x4_t v = {0.0, 0.0, 0.0, 0.0};
    vmlaq_f32(v, v, v);
  }
  ]] HAVE_NEON)

if (NOT HAVE_AVX2 AND NOT HAVE_NEON)
  message(FATAL_ERROR "cannot build without one of AVX2 or neon")
endif()

if (NOT DEFINED TARGET_ARCH)
  message(FATAL_ERROR "TARGET_ARCH is not defined by the preset configuration")
endif()

find_package(Exo REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
