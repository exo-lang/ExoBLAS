from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def iamax_template(
  n: size,
  x: [R][n],
  index: i32
):
    maxVal: R
    maxVal = x[0]
    index = 0.0
    counter: i32
    counter = 0.0
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        index = select(maxVal, xReg, counter, index)
        maxVal = select(maxVal, xReg, xReg, maxVal)
        counter += 1.0
        
def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(iamax_template, "exo_i" + prefix + "amax")
    for arg in ["x", "maxVal", "xReg"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy

def schedule_iamax_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")

    return simple_stride_1

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_isamax_stride_any = specialize_precision("f32")
exo_isamax_stride_any = rename(exo_isamax_stride_any, exo_isamax_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     ]
if None not in f32_instructions:
    exo_isamax_stride_1 = schedule_iamax_stride_1(C.Machine.vec_width, C.Machine.mem_type, f32_instructions, "f32")
else:
    exo_isamax_stride_1 = specialize_precision("f32")
    exo_isamax_stride_1 = rename(exo_isamax_stride_1, exo_isamax_stride_1.name() + "_stride_1")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_idamax_stride_any = specialize_precision("f64")
exo_idamax_stride_any = rename(exo_idamax_stride_any, exo_idamax_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     ]

if None not in f64_instructions:
    exo_idamax_stride_1 = schedule_iamax_stride_1(C.Machine.vec_width // 2, C.Machine.mem_type, f64_instructions, "f64")
else:
    exo_idamax_stride_1 = specialize_precision("f64")
    exo_idamax_stride_1 = rename(exo_idamax_stride_1, exo_idamax_stride_1.name() + "_stride_1")

@proc
def exo_isamax_stride_any_f32(n: size, x: [f32][n] @ DRAM, index: i32 @ DRAM):
    maxVal: f32 @ DRAM
    maxVal = x[0]
    index = 0.0
    counter: i32 @ DRAM
    counter = 0.0
    for i in seq(0, n):
        xReg: f32 @ DRAM
        xReg = x[i]
        index = select(maxVal, xReg, counter, index)
        maxVal = select(maxVal, xReg, xReg, maxVal)
        counter += 1.0

entry_points = [exo_isamax_stride_any_f32]

for p in entry_points:
    print(p)

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
