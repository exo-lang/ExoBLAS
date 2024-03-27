#pragma once

#include <cblas.h>

#include <string>

std::pair<int, int> get_dims(const enum CBLAS_TRANSPOSE trans, int M, int N,
                             int ld_diff) {
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    return {M, N + ld_diff};
  } else {
    return {N, M + ld_diff};
  }
}

class BLAS_lib {};

class Exo : public BLAS_lib {
 public:
  static std::string lib_name() { return "exo"; }
};

class Cblas : public BLAS_lib {
 public:
  static std::string lib_name() { return "cblas"; }
};

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

std::string order_symbol(int Order) {
  if (Order == CBLAS_ORDER::CblasRowMajor) {
    return "_rm";
  } else if (Order == CBLAS_ORDER::CblasColMajor) {
    return "_col";
  } else {
    return "";
  }
}

std::string side_symbol(int Side) {
  if (Side == CBLAS_SIDE::CblasLeft) {
    return "l";
  } else if (Side == CBLAS_SIDE::CblasRight) {
    return "r";
  } else {
    return "";
  }
}

std::string trans_symbol(int Trans) {
  if (Trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    return "n";
  } else if (Trans == CBLAS_TRANSPOSE::CblasTrans) {
    return "t";
  } else {
    return "";
  }
}

std::string uplo_symbol(int Uplo) {
  if (Uplo == CBLAS_UPLO::CblasLower) {
    return "l";
  } else if (Uplo == CBLAS_UPLO::CblasUpper) {
    return "u";
  } else {
    return "";
  }
}

template <typename lib, typename T>
std::string kernel_name(std::string kernel) {
  return lib::lib_name() + "_" + type_prefix<T>() + kernel;
}

template <typename lib, typename T, int order, int TransA, int Uplo>
std::string level_2_kernel_name(std::string kernel) {
  auto name = lib::lib_name() + "_" + type_prefix<T>() + kernel;
  name += order_symbol(order);
  name += Uplo + TransA ? "_" : "";
  name += uplo_symbol(Uplo);
  name += trans_symbol(TransA);
  return name;
}

template <typename lib, typename T>
std::string level_3_kernel_name(std::string kernel, int Order, int Side,
                                int Uplo, int TransA, int TransB) {
  auto name = lib::lib_name() + "_" + type_prefix<T>() + kernel;
  name += order_symbol(Order);
  name += Side + Uplo + TransA + TransB ? "_" : "";
  name += side_symbol(Side);
  name += uplo_symbol(Uplo);
  name += trans_symbol(TransA);
  name += trans_symbol(TransB);
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
