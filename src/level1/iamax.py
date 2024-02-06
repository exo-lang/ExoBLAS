from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


### EXO_LOC ALGORITHM START ###
@proc
def exo_isamax(n: size, x: [f32][n], index: i32):
    maxVal: f32
    maxVal = x[0]
    index = 0.0
    counter: i32
    counter = 0.0
    for i in seq(0, n):
        xReg: f32
        xReg = x[i]
        xAbs: f32
        negX: f32
        negX = -xReg
        zero: f32
        zero = 0.0
        xAbs = select(zero, xReg, xReg, negX)
        index = select(maxVal, xAbs, counter, index)
        maxVal = select(maxVal, xAbs, xAbs, maxVal)
        counter += 1.0


@proc
def exo_idamax(n: size, x: [f64][n], index: i32):
    maxVal: f64
    maxVal = -1.0
    index = 0.0
    counter: i32
    counter = 0.0
    for i in seq(0, n):
        xReg: f64
        xReg = x[i]
        xAbs: f64
        negX: f64
        negX = -xReg
        zero: f64
        zero = 0.0
        xAbs = select(zero, xReg, xReg, negX)
        index = select(maxVal, xAbs, counter, index)
        maxVal = select(maxVal, xAbs, xAbs, maxVal)
        counter += 1.0


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_iamax_stride_1(iamax, VEC_W, memory, instructions, precision):
    simple_stride_1 = rename(iamax, iamax.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")

    return simple_stride_1


#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_isamax_stride_any = exo_isamax
exo_isamax_stride_any = rename(
    exo_isamax_stride_any, exo_isamax_stride_any.name() + "_stride_any"
)

f32_instructions = []
if None not in f32_instructions:
    exo_isamax_stride_1 = schedule_iamax_stride_1(
        exo_isamax, C.Machine.f32_vec_width, C.Machine.mem_type, f32_instructions, "f32"
    )
else:
    exo_isamax_stride_1 = exo_isamax
    exo_isamax_stride_1 = rename(
        exo_isamax_stride_1, exo_isamax_stride_1.name() + "_stride_1"
    )

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_idamax_stride_any = exo_idamax
exo_idamax_stride_any = rename(
    exo_idamax_stride_any, exo_idamax_stride_any.name() + "_stride_any"
)

f64_instructions = []

if None not in f64_instructions:
    exo_idamax_stride_1 = schedule_iamax_stride_1(
        exo_idamax,
        C.Machine.f32_vec_width // 2,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
else:
    exo_idamax_stride_1 = exo_idamax
    exo_idamax_stride_1 = rename(
        exo_idamax_stride_1, exo_idamax_stride_1.name() + "_stride_1"
    )
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_isamax_stride_any,
    exo_isamax_stride_1,
    exo_idamax_stride_any,
    exo_idamax_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
