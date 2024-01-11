#include "exo_sad.h"



#include <stdio.h>
#include <stdlib.h>

double _select_(double x, double v, double y, double z) {
    if (x < v) return y;
    else return z;
}


// sad(
//     x : i8[1024] @DRAM,
//     y : i8[1024] @DRAM,
//     result : f32 @DRAM
// )
void sad( void *ctxt, const int8_t* x, const int8_t* y, float* result ) {
*result = 0.0;
float *var0 = (float*) malloc(4 * sizeof(*var0));
for (int_fast32_t ioi = 0; ioi < 4; ioi++) {
  var0[ioi] = 0.0;
}
for (int_fast32_t ioo = 0; ioo < 32; ioo++) {
  float *tmp_reg1 = (float*) malloc(4 * sizeof(*tmp_reg1));
  int8_t *xReg = (int8_t*) malloc(32 * sizeof(*xReg));
  for (int_fast32_t i0 = 0; i0 < 32; i0++) {
    xReg[i0] = x[i0 + 32 * ioo];
  }
  int8_t *yReg = (int8_t*) malloc(32 * sizeof(*yReg));
  for (int_fast32_t i0 = 0; i0 < 32; i0++) {
    yReg[i0] = y[i0 + 32 * ioo];
  }
  for (int_fast32_t ioi = 0; ioi < 4; ioi++) {
    float tmp_reg;
    tmp_reg = 0.0;
    for (int_fast32_t ii = 0; ii < 8; ii++) {
      int8_t arg;
      arg = yReg[ii + 8 * ioi];
      int8_t arg_1;
      arg_1 = xReg[ii + 8 * ioi];
      int8_t arg_2;
      arg_2 = xReg[ii + 8 * ioi] - yReg[ii + 8 * ioi];
      int8_t arg_3;
      arg_3 = yReg[ii + 8 * ioi] - xReg[ii + 8 * ioi];
      tmp_reg += (float)(_select_((double)*&arg, (double)*&arg_1, (double)*&arg_2, (double)*&arg_3));
    }
    tmp_reg1[ioi] = tmp_reg;
  }
  free(yReg);
  free(xReg);
  for (int_fast32_t ioi = 0; ioi < 4; ioi++) {
    var0[ioi] += tmp_reg1[ioi];
  }
  free(tmp_reg1);
}
for (int_fast32_t ioi = 0; ioi < 4; ioi++) {
  *result += var0[ioi];
}
free(var0);
}

