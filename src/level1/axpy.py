from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def axpy_template(
  n: size,
  alpha: R,
  x: [R][n],
  y: [R][n],
):
    for i in seq(0, n):
        y[i] += alpha * x[i]
        
@proc
def axpy_template_alpha_1(
  n: size,
  x: [R][n],
  y: [R][n],
):
    for i in seq(0, n):
        y[i] += x[i]
        
def specialize_axpy(precision, alpha):
    prefix = "s" if precision == "f32" else "d"
    specialized_axpy = axpy_template if alpha != 1 else axpy_template_alpha_1
    axpy_template_name = specialized_axpy.name()
    axpy_template_name = axpy_template_name.replace("_template", "")
    specialized_axpy = rename(specialized_axpy, "exo_" + prefix + axpy_template_name)
    
    args = ["x", "y"]
    if alpha != 1:
        args.append("alpha")
        
    for arg in args:
        specialized_axpy = set_precision(specialized_axpy, arg, precision)
    return specialized_axpy

def schedule_interleave_axpy_stride_1(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision, alpha):
    simple_stride_1 = specialize_axpy(precision, alpha)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    def stage_expr(proc, expr, buffer):
        proc = bind_expr(proc, expr, buffer)
        proc = expand_dim(proc, buffer, VEC_W, "ii")
        proc = lift_alloc(proc, f"{buffer}:_", n_lifts=1)
        proc = fission(proc, proc.find(f"{buffer}[_] = _").after())
        return proc

    loop_fragment = lambda iter, idx=0: f"for {iter} in _:_ #{idx}"
    simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("i"), VEC_W, ("io", "ii"), tail="cut")

    registers = ["xReg", "yReg"]
    
    if alpha != 1:
        simple_stride_1 = stage_expr(simple_stride_1, "alpha", "alphaReg")
        registers.append("alphaReg")

    simple_stride_1 = stage_expr(simple_stride_1, "x[_]", "xReg")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, simple_stride_1.find("y[_] += _").parent(), f"y[io * {VEC_W}:(io + 1) * {VEC_W}]", "yReg"))
    
    for buffer in registers:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    def hoist_const_broadcast(proc, constant):
        while True:
            try:
                call_cursor = proc.find(constant).parent()
                proc = reorder_stmts(proc, call_cursor.expand(-1))
            except:
                break
        call_cursor = proc.find(constant).parent()
        proc = autofission(proc, call_cursor.after(), n_lifts=2)
        return proc

    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for im in _:_")
        return proc
    
    if INTERLEAVE_FACTOR > 1:
        simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("io"), INTERLEAVE_FACTOR, ("io", "im"), tail="cut")
        
        for reg in registers:
            simple_stride_1 = expand_dim(simple_stride_1, reg, INTERLEAVE_FACTOR, "im")
            simple_stride_1 = lift_alloc(simple_stride_1, reg, n_lifts=2)
            
        if alpha != 1:
            simple_stride_1 = lift_alloc(simple_stride_1, "alphaReg #1", n_lifts=1) # Tail loop
        
        simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    else:
        if alpha != 1:
            simple_stride_1 = lift_alloc(simple_stride_1, "alphaReg", n_lifts=1)
    
    if alpha != 1:
        for i in range(INTERLEAVE_FACTOR + 1):
            simple_stride_1 = hoist_const_broadcast(simple_stride_1, f"alpha #{i}")
    
    return simplify(simple_stride_1)

#################################################
# Generate specialized kernels for f32 precision
#################################################

INTERLEAVE_FACTOR = 4

exo_saxpy_stride_any = specialize_axpy("f32", None)
exo_saxpy_stride_any = rename(exo_saxpy_stride_any, exo_saxpy_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     C.Machine.reduce_add_wide_instr_f32,
                     ]
if None not in f32_instructions:
    exo_saxpy_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width, INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32", None)
    exo_saxpy_alpha_1_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width, INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32", 1)
else:
    exo_saxpy_stride_1 = specialize_axpy("f32", None)
    exo_saxpy_stride_1 = rename(exo_saxpy_stride_1, exo_saxpy_stride_1.name() + "_stride_1")
    exo_saxpy_alpha_1_stride_1 = specialize_axpy("f32", 1)
    exo_saxpy_alpha_1_stride_1 = rename(exo_saxpy_alpha_1_stride_1, exo_saxpy_alpha_1_stride_1.name() + "_stride_1")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_daxpy_stride_any = specialize_axpy("f64", None)
exo_daxpy_stride_any = rename(exo_daxpy_stride_any, exo_daxpy_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     C.Machine.reduce_add_wide_instr_f64,
                     ]

if None not in f64_instructions:
    exo_daxpy_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width // 2, INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64", None)
    exo_daxpy_alpha_1_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width // 2, INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64", 1)
else:
    exo_daxpy_stride_1 = specialize_axpy("f64", None)
    exo_daxpy_stride_1 = rename(exo_daxpy_stride_1, exo_daxpy_stride_1.name() + "_stride_1")
    exo_daxpy_alpha_1_stride_1 = specialize_axpy("f64", 1)
    exo_daxpy_alpha_1_stride_1 = rename(exo_daxpy_alpha_1_stride_1, exo_daxpy_alpha_1_stride_1.name() + "_stride_1")

entry_points = [exo_saxpy_stride_any, exo_saxpy_stride_1, exo_saxpy_alpha_1_stride_1,
                exo_daxpy_stride_any, exo_daxpy_stride_1, exo_daxpy_alpha_1_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
