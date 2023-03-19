from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def ger_raw_major_template(m: size, n: size, alpha: R, x: [R][m], y: [R][n], A: [R][m, n]):
    assert stride(A, 1) == 1
    
    for i in seq(0, m):
        for j in seq(0, n):
            A[i, j] += alpha * x[i] * y[j]

def specialize_ger(precision):
    prefix = "s" if precision == "f32" else "d"
    name = ger_raw_major_template.name()
    name = name.replace("_template", "")
    specialized = rename(ger_raw_major_template, "exo_" + prefix + name)
    
    args = ["alpha", "x", "y", "A"]
        
    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized

def schedule_interleave_ger_raw_major_stride_1(VEC_W, RAW_INTERLEAVE_FACTOR, memory, instructions, precision):
    stride_1 = specialize_ger(precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
    
    stride_1 = stage_mem(stride_1, stride_1.body(), "alpha", "alpha_")
    
    return stride_1    
    
    stride_1 = divide_loop(stride_1, stride_1.find_loop("j"), VEC_W, ("jo", "ji"), tail="cut")
    
    def stage_expr(proc, expr, buffer):
        proc = bind_expr(proc, expr, buffer)
        proc = expand_dim(proc, buffer, VEC_W, "ji")
        proc = lift_alloc(proc, f"{buffer}:_", n_lifts=1)
        proc = fission(proc, proc.find(f"{buffer}[_] = _").after())
        return proc
    
    stride_1 = stage_expr(stride_1, [stride_1.find("alpha_ #0")], "alphaReg")
    stride_1 = stage_expr(stride_1, [stride_1.find("x[_]")], "xReg")
    stride_1 = stage_expr(stride_1, [stride_1.find("alphaReg[_] * xReg[_]")], "xMulAlphaReg")
    stride_1 = stage_expr(stride_1, [stride_1.find("y[_]")], "yReg")
    lower_bound = f"{VEC_W} * jo"
    stride_1 = stage_mem(stride_1, stride_1.find_loop("ji #4"), f"A[i, {lower_bound}:{lower_bound} + {VEC_W}]", "AReg")
    
    registers = ["alphaReg", "xReg", "xMulAlphaReg", "yReg", "AReg"]
    
    for reg in registers:
        stride_1 = set_memory(stride_1, reg, memory)
        stride_1 = set_precision(stride_1, reg, precision)
    
    stride_1 = replace_all(stride_1, instructions)
    
    stride_1 = divide_loop(stride_1, stride_1.find_loop("i"), RAW_INTERLEAVE_FACTOR, ("io", "ii"), tail="cut")
    stride_1 = fission(stride_1, stride_1.find_loop("jo").after())
    stride_1 = reorder_loops(stride_1, "ii jo")
    stride_1 = unroll_loop(stride_1, "ii")
    
    stride_1 = stage_mem(stride_1, stride_1.find_loop("ji"), f"x[ii + {RAW_INTERLEAVE_FACTOR} * io]", "xVal")
    stride_1 = stage_mem(stride_1, stride_1.find_loop("ji #1"), f"x[ii + m / {RAW_INTERLEAVE_FACTOR} * {RAW_INTERLEAVE_FACTOR}]", "xVal")
    
    return simplify(stride_1)

#################################################
# Kernel Parameters
#################################################

RAW_INTERLEAVE_FACTOR = C.Machine.vec_units

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sger_raw_major_stride_any = specialize_ger("f32")
exo_sger_raw_major_stride_any = rename(exo_sger_raw_major_stride_any, exo_sger_raw_major_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.mul_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     C.Machine.broadcast_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     ]

exo_sger_raw_major_stride_1 = schedule_interleave_ger_raw_major_stride_1(C.Machine.vec_width, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dger_raw_major_stride_any = specialize_ger("f64")
exo_dger_raw_major_stride_any = rename(exo_dger_raw_major_stride_any, exo_dger_raw_major_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.mul_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     C.Machine.broadcast_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     ]

exo_dger_raw_major_stride_1 = schedule_interleave_ger_raw_major_stride_1(C.Machine.vec_width // 2, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64")

entry_points = [exo_sger_raw_major_stride_any, exo_sger_raw_major_stride_1,
                exo_dger_raw_major_stride_any, exo_dger_raw_major_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
