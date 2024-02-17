#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>

#include "misc.h"

double constexpr default_error = 1.0 / 100.0;

bool check_relative_error_okay(double result, double expected,
                               double epsilon = default_error) {
  auto error = std::fabs(result - expected);
  auto rel_error = error / std::max((double)1.0, std::fabs(expected));
  return rel_error < epsilon;
}

template <typename Tname, typename Tvalue, typename... Args>
void print_params_(Tname&& name, Tvalue&& value, Args&&... args) {
  std::cout << name << " = " << value;
  if constexpr (sizeof...(args)) {
    std::cout << ", ";
    print_params_(args...);
  } else {
    std::cout << std::endl;
  }
}

template <typename... Args>
void print_params(Args&&... args) {
  std::cout << "Params: ";
  print_params_(args...);
}

template <typename T, typename... Args>
void failed(std::string&& kernel, Args&&... args) {
  std::cout << "Running " << kernel_name<Exo, T>(kernel) << std::endl;
  print_params(args...);
  std::cout << "Failed!" << std::endl;
  exit(1);
}
