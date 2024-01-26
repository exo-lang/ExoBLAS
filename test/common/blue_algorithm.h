#pragma once

#include <limits.h>

#include <cmath>

/**
Based on:

Anderson E. (2017)
Algorithm 978: Safe Scaling in the Level 1 BLAS
ACM Trans Math Softw 44:1--28
https://doi.org/10.1145/3061665

Blue, James L. (1978)
A Portable Fortran Program to Find the Euclidean Norm of a Vector
ACM Trans Math Softw 4:15--23
https://doi.org/10.1145/355769.355771
*/

template <typename T>
class blue_algorithm {
 public:
  static constexpr T t_sml =
      std::pow(std::numeric_limits<T>::radix,
               std::ceil((std::numeric_limits<T>::min_exponent - 1) / 2.0));

  static constexpr T t_big =
      std::pow(std::numeric_limits<T>::radix,
               std::floor((std::numeric_limits<T>::max_exponent -
                           std::numeric_limits<T>::digits + 1) /
                          2.0));

  static constexpr T s_sml =
      std::pow(std::numeric_limits<T>::radix,
               -std::floor((std::numeric_limits<T>::min_exponent -
                            std::numeric_limits<T>::digits) /
                           2.0));

  static constexpr T s_big =
      std::pow(std::numeric_limits<T>::radix,
               -std::ceil((std::numeric_limits<T>::max_exponent -
                           std::numeric_limits<T>::digits + 1) /
                          2.0));

  static T combine_accums(T a_sml, T a_med, T a_big) {
    T q, w;
    if (a_big > 0) {
      if (a_med > 0 || std::isnan(a_med)) {
        a_big += (a_med * s_big) * s_big;
      }
      w = 1.0 / s_big;
      q = a_big;
    } else if (a_sml > 0) {
      if (a_med > 0 || std::isnan(a_med)) {
        T y1 = std::sqrt(a_med);
        T y2 = std::sqrt(a_sml) / s_sml;
        T y_min = std::min(y1, y2);
        T y_max = std::max(y1, y2);
        w = 1;
        T min_max_ratio = y_min / y_max;
        q = (y_max * y_max) * (1 + min_max_ratio * min_max_ratio);
      } else {
        w = 1 / s_sml;
        q = a_sml;
      }
    } else {
      w = 1.0;
      q = a_med;
    }
    return w * std::sqrt(q);
  }
};
