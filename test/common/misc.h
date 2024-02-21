#pragma once

#include <cblas.h>

#include <string>

class BLAS_lib {};

class Exo : public BLAS_lib {};

class Cblas : public BLAS_lib {};

template <typename lib>
std::string lib_name() {
  if constexpr (std::is_same<lib, Exo>::value) {
    return "exo";
  } else {
    return "cblas";
  }
}

template <typename T>
std::string type_prefix() {
  if constexpr (std::is_same<T, float>::value) {
    return "s";
  } else {
    return "d";
  }
}

template <typename T>
int type_bits() {
  if constexpr (std::is_same<T, float>::value) {
    return 32;
  } else {
    return 64;
  }
}

template <typename lib, typename T>
std::string kernel_name(std::string kernel) {
  return lib_name<lib>() + "_" + type_prefix<T>() + kernel;
}

template <typename lib, typename T, int order, int TransA>
std::string level_2_kernel_name(std::string kernel) {
  auto name = lib_name<lib>() + "_" + type_prefix<T>() + kernel;
  name += order == CBLAS_ORDER::CblasRowMajor ? "_rm" : "_col";
  name += TransA == CBLAS_TRANSPOSE::CblasNoTrans ? "_n" : "_t";
  return name;
}

#define call_bench(lib, T, kernel) \
  BENCHMARK(bench<lib, T>)->Name(kernel_name<lib, T>(#kernel))->Apply(args<T>);
#define call_bench_all(kernel)      \
  call_bench(Exo, float, kernel);   \
  call_bench(Exo, double, kernel);  \
  call_bench(Cblas, float, kernel); \
  call_bench(Cblas, double, kernel);

#define generate_wrapper_ret(kernel)                    \
  template <typename lib, typename T, typename... Args> \
  T kernel(Args... args) {                              \
    if constexpr (std::is_same<T, float>::value) {      \
      if constexpr (std::is_same<lib, Exo>::value) {    \
        return exo_s##kernel(args...);                  \
      } else {                                          \
        return cblas_s##kernel(args...);                \
      }                                                 \
    } else {                                            \
      if constexpr (std::is_same<lib, Exo>::value) {    \
        return exo_d##kernel(args...);                  \
      } else {                                          \
        return cblas_d##kernel(args...);                \
      }                                                 \
    }                                                   \
  }

#define generate_wrapper(kernel)                        \
  template <typename lib, typename T, typename... Args> \
  void kernel(Args... args) {                           \
    if constexpr (std::is_same<T, float>::value) {      \
      if constexpr (std::is_same<lib, Exo>::value) {    \
        exo_s##kernel(args...);                         \
      } else {                                          \
        cblas_s##kernel(args...);                       \
      }                                                 \
    } else {                                            \
      if constexpr (std::is_same<lib, Exo>::value) {    \
        exo_d##kernel(args...);                         \
      } else {                                          \
        cblas_d##kernel(args...);                       \
      }                                                 \
    }                                                   \
  }\
