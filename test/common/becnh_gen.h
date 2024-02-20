#include <tuple>

class BLAS_Param {};

class Int : public BLAS_Param {};

class Size : public Int {};

class Inc : public Int {};

class Alignment : public Int {};

template <typename T>
class Const : public BLAS_Param {};

template <typename T>
class Buffer : public BLAS_Param {};

template <typename T, unsigned int Size_index, unsigned int Inc_index,
          unsigned int Alignment_Index>
class 1D_Buffer : public Buffer<T>{

                  };

template <typename First, typename... Args>
void check_sig() {
  static_assert(std::is_base_of<BLAS_Param, First>::value);
  if constexpr (sizeof...(Args)) {
    check_sig<Args...>();
  }
}

auto generate_init_tuple(benchmark::State &state, int idx) {
  return std::tuple<>();
}

template <unsigned int idx>
auto generate_init_tuple(Int, benchmark::State &state) {
  return std::tuple<int>(state.range(idx));
}

template <unsigned int idx, typename T>
auto generate_init_tuple(Const<T>, benchmark::State &state) {
  return std::tuple<T>(state.range(idx));
}

template <unsigned int idx, typename T>
auto generate_init_tuple(Buffer<T>, benchmark::State &state) {
  return std::tuple<T *>(nullptr);
}

template <unsigned int idx, typename First, typename... Args>
auto generate_init_tuple(benchmark::State &state) {
  auto other = generate_init_tuple<idx + 1, Args...>(state);
  First obj;
  auto first = generate_init_tuple<idx + 1>(obj, state);
  return std::tuple_cat(first, other);
}

template <typename Tuple, typename T, unsigned int Size_index,
          unsigned int Inc_index, unsigned int Alignment_Index>
auto generate_final_tuple(Buffer<T, Size_index, Inc_index, Alignment_Index>,
                          Tuple &&init_args) {
  int size = std::get<Size_index>(init_args);
  int incX = std::get<Inc_index>(init_args);
  int align = std::get<Alignment_Index>(init_args);
  return std::tuple<AlignedBuffer>()
}

template <typename Tuple, typename First, typename... Args>
auto generate_final_tuple(Tuple &&init_args) {
  auto other = generate_final_tuple<Args...>(state, idx + 1);
  First obj;
  auto first = generate_final_tuple(obj, state, idx + 1);
  return std::tuple_cat(first, other);
}

template <typename lib, typename T, typename... Sig>
static void bench(benchmark::State &state) {
  check_sig<Sig...>();
  auto init_tuple = generate_init_tuple<Sig...>(state, 0);

  int N = state.range(0);
  int incX = state.range(1);
  size_t alignmentX = state.range(2);
  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  for (auto _ : state) {
    asum<lib, T>(N, X.data(), incX);
  }
}
