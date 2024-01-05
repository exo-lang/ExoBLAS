#include "exo_iamax.h"



#include <stdio.h>
#include <stdlib.h>

double _select_(double x, double v, double y, double z) {
    if (x < v) return y;
    else return z;
}


// exo_idamax_stride_1(
//     n : size,
//     x : [f64][n] @DRAM,
//     index : i32 @DRAM
// )
void exo_idamax_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, int32_t* index ) {
// assert stride(x, 0) == 1
double maxVal;
maxVal = -1.0;
*index = 0.0;
int32_t counter;
counter = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i];
  double xAbs;
  double negX;
  negX = -xReg;
  double zero;
  zero = 0.0;
  xAbs = _select_((double)*&zero, (double)*&xReg, (double)*&xReg, (double)*&negX);
  *index = _select_((double)*&maxVal, (double)*&xAbs, (double)*&counter, (double)*index);
  maxVal = _select_((double)*&maxVal, (double)*&xAbs, (double)*&xAbs, (double)*&maxVal);
  counter += 1.0;
}
}

// exo_idamax_stride_any(
//     n : size,
//     x : [f64][n] @DRAM,
//     index : i32 @DRAM
// )
void exo_idamax_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f64c x, int32_t* index ) {
double maxVal;
maxVal = -1.0;
*index = 0.0;
int32_t counter;
counter = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  double xReg;
  xReg = x.data[i * x.strides[0]];
  double xAbs;
  double negX;
  negX = -xReg;
  double zero;
  zero = 0.0;
  xAbs = _select_((double)*&zero, (double)*&xReg, (double)*&xReg, (double)*&negX);
  *index = _select_((double)*&maxVal, (double)*&xAbs, (double)*&counter, (double)*index);
  maxVal = _select_((double)*&maxVal, (double)*&xAbs, (double)*&xAbs, (double)*&maxVal);
  counter += 1.0;
}
}

// exo_isamax_stride_1(
//     n : size,
//     x : [f32][n] @DRAM,
//     index : i32 @DRAM
// )
void exo_isamax_stride_1( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, int32_t* index ) {
// assert stride(x, 0) == 1
float maxVal;
maxVal = x.data[0];
*index = 0.0;
int32_t counter;
counter = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i];
  float xAbs;
  float negX;
  negX = -xReg;
  float zero;
  zero = 0.0;
  xAbs = _select_((double)*&zero, (double)*&xReg, (double)*&xReg, (double)*&negX);
  *index = _select_((double)*&maxVal, (double)*&xAbs, (double)*&counter, (double)*index);
  maxVal = _select_((double)*&maxVal, (double)*&xAbs, (double)*&xAbs, (double)*&maxVal);
  counter += 1.0;
}
}

// exo_isamax_stride_any(
//     n : size,
//     x : [f32][n] @DRAM,
//     index : i32 @DRAM
// )
void exo_isamax_stride_any( void *ctxt, int_fast32_t n, struct exo_win_1f32c x, int32_t* index ) {
float maxVal;
maxVal = x.data[0];
*index = 0.0;
int32_t counter;
counter = 0.0;
for (int_fast32_t i = 0; i < n; i++) {
  float xReg;
  xReg = x.data[i * x.strides[0]];
  float xAbs;
  float negX;
  negX = -xReg;
  float zero;
  zero = 0.0;
  xAbs = _select_((double)*&zero, (double)*&xReg, (double)*&xReg, (double)*&negX);
  *index = _select_((double)*&maxVal, (double)*&xAbs, (double)*&counter, (double)*index);
  maxVal = _select_((double)*&maxVal, (double)*&xAbs, (double)*&xAbs, (double)*&maxVal);
  counter += 1.0;
}
}

