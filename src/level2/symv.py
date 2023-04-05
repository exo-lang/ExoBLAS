from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def symv_scal_y(n: size,
                beta: R,
                y: [R][n]):
    
    for i in seq(0, n):
        y[i] = beta * y[i]

@proc
def symv_raw_major_Upper_template(n: size, 
                                  alpha: R,
                                  A: [R][n, n],
                                  x: [R][n], 
                                  y: [R][n]):
    assert stride(A, 1) == 1
    
    for i in seq(0, n):
        temp: R
        temp = alpha * x[i]
        dot: R
        dot = 0.0
        for j in seq(0, n - i - 1):
            y[i + j + 1] += temp * A[i, i + j + 1]
            dot += A[i, i + j + 1] * x[i + j + 1]
        y[i] += temp * A[i, i] + alpha * dot

@proc
def symv_raw_major_Lower_template(n: size, 
                                  alpha: R,
                                  A: [R][n, n],
                                  x: [R][n], 
                                  y: [R][n]):
    assert stride(A, 1) == 1
    
    for i in seq(0, n):
        temp: R
        temp = alpha * x[i]
        dot: R
        dot = 0.0
        for j in seq(0, i):
            y[j] += temp * A[i, j]
            dot += A[i, j] * x[j]
        y[i] += temp * A[i, i] + alpha * dot

def specialize_symv(symv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = symv.name()
    name = name.replace("_template", "")
    specialized = rename(symv, "exo_" + prefix + name)
    
    if "scal" in symv.name():
        args = ["y", "beta"]
    else:
        args = ["x", "A", "alpha", "y", "temp", "dot"]
        
    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized

def schedule_interleave_symv_scal_y(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    stride_1 = specialize_symv(symv_scal_y, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
    
    return stride_1
    
def schedule_interleave_symv_raw_major_stride_1(symv, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    stride_1 = specialize_symv(symv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
    
    return stride_1
    
#################################################
# Kernel Parameters
#################################################

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_ssymv_scal_y_stride_any = specialize_symv(symv_scal_y, "f32")
exo_ssymv_scal_y_stride_any = rename(exo_ssymv_scal_y_stride_any, 
                                        exo_ssymv_scal_y_stride_any.name() + "_stride_any")
exo_ssymv_raw_major_Upper_stride_any = specialize_symv(symv_raw_major_Upper_template, "f32")
exo_ssymv_raw_major_Upper_stride_any = rename(exo_ssymv_raw_major_Upper_stride_any, 
                                            exo_ssymv_raw_major_Upper_stride_any.name() + "_stride_any")
exo_ssymv_raw_major_Lower_stride_any = specialize_symv(symv_raw_major_Lower_template, "f32")
exo_ssymv_raw_major_Lower_stride_any = rename(exo_ssymv_raw_major_Lower_stride_any, 
                                            exo_ssymv_raw_major_Lower_stride_any.name() + "_stride_any")
f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.mul_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     C.Machine.broadcast_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     ]

exo_ssymv_raw_major_Upper_stride_1 = schedule_interleave_symv_raw_major_stride_1(symv_raw_major_Upper_template,
                                                                                           C.Machine.vec_width, 1, C.Machine.mem_type, f32_instructions, "f32")
exo_ssymv_raw_major_Lower_stride_1 = schedule_interleave_symv_raw_major_stride_1(symv_raw_major_Lower_template,
                                                                                           C.Machine.vec_width, 1, C.Machine.mem_type, f32_instructions, "f32")
exo_ssymv_scal_y_stride_1 = schedule_interleave_symv_scal_y(C.Machine.vec_width, 1, C.Machine.mem_type, f32_instructions, "f32")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dsymv_scal_y_stride_any = specialize_symv(symv_scal_y, "f64")
exo_dsymv_scal_y_stride_any = rename(exo_dsymv_scal_y_stride_any, 
                                        exo_dsymv_scal_y_stride_any.name() + "_stride_any")
exo_dsymv_raw_major_Upper_stride_any = specialize_symv(symv_raw_major_Upper_template, "f64")
exo_dsymv_raw_major_Upper_stride_any = rename(exo_dsymv_raw_major_Upper_stride_any,
                                                        exo_dsymv_raw_major_Upper_stride_any.name() + "_stride_any")
exo_dsymv_raw_major_Lower_stride_any = specialize_symv(symv_raw_major_Lower_template, "f64")
exo_dsymv_raw_major_Lower_stride_any = rename(exo_dsymv_raw_major_Lower_stride_any,
                                                        exo_dsymv_raw_major_Lower_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.mul_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     C.Machine.broadcast_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     ]

exo_dsymv_raw_major_Upper_stride_1 = schedule_interleave_symv_raw_major_stride_1(symv_raw_major_Upper_template,
                                                                                           C.Machine.vec_width // 2, 1, C.Machine.mem_type, f64_instructions, "f64")
exo_dsymv_raw_major_Lower_stride_1 = schedule_interleave_symv_raw_major_stride_1(symv_raw_major_Lower_template,
                                                                                           C.Machine.vec_width // 2, 1, C.Machine.mem_type, f64_instructions, "f64")
exo_dsymv_scal_y_stride_1 = schedule_interleave_symv_scal_y(C.Machine.vec_width // 2, 1, C.Machine.mem_type, f64_instructions, "f64")

entry_points = [
                exo_ssymv_scal_y_stride_any, exo_ssymv_scal_y_stride_1,
                exo_ssymv_raw_major_Upper_stride_any, exo_ssymv_raw_major_Upper_stride_1,
                exo_dsymv_raw_major_Upper_stride_any, exo_dsymv_raw_major_Upper_stride_1,
                
                exo_ssymv_raw_major_Lower_stride_any, exo_ssymv_raw_major_Lower_stride_1,
                exo_dsymv_raw_major_Lower_stride_any, exo_dsymv_raw_major_Lower_stride_1,
                exo_dsymv_scal_y_stride_any, exo_dsymv_scal_y_stride_1,
                ]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
