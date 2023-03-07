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
        
def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(axpy_template, "exo_" + prefix + "axpy")
    for arg in ["x", "y", "alpha"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy

def schedule_axpy_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    def stage_expr(proc, expr, buffer):
        proc = bind_expr(proc, expr, buffer)
        proc = expand_dim(proc, buffer, VEC_W, "ii")
        proc = lift_alloc(proc, f"{buffer}:_", n_lifts=2)
        proc = fission(proc, proc.find(f"{buffer}[_] = _").after())
        return proc
    
    def hoist_const_loop(proc, constant):
        while True:
            try:
                loop_cursor = proc.find(constant).parent().parent()
                proc = reorder_stmts(proc, loop_cursor.expand(-1))
            except:
                break
        loop_cursor = proc.find(constant).parent().parent()
        proc = autofission(proc, loop_cursor.after())
        return proc

    loop_fragment = lambda iter, idx=0: f"for {iter} in _:_ #{idx}"
    simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("i"), VEC_W, ("io", "ii"), tail="cut")

    simple_stride_1 = stage_expr(simple_stride_1, "alpha", "alphaReg")
    simple_stride_1 = stage_expr(simple_stride_1, "x[_]", "xReg")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, loop_fragment("ii", 2), f"y[io * {VEC_W}:(io + 1) * {VEC_W}]", "yReg"))
    
    simple_stride_1 = hoist_const_loop(simple_stride_1, "alpha")
    
    for buffer in ["xReg", "alphaReg", "yReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simplify(simple_stride_1)

def schedule_interleave_axpy_stride_1(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    def stage_expr(proc, expr, buffer):
        proc = bind_expr(proc, expr, buffer)
        proc = expand_dim(proc, buffer, VEC_W, "ii")
        proc = lift_alloc(proc, f"{buffer}:_", n_lifts=1)
        proc = fission(proc, proc.find(f"{buffer}[_] = _").after())
        return proc
    
    def hoist_const_loop(proc, constant):
        while True:
            try:
                loop_cursor = proc.find(constant).parent().parent()
                proc = reorder_stmts(proc, loop_cursor.expand(-1))
            except:
                break
        loop_cursor = proc.find(constant).parent().parent()
        proc = autofission(proc, loop_cursor.after())
        return proc

    loop_fragment = lambda iter, idx=0: f"for {iter} in _:_ #{idx}"
    simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("i"), VEC_W * INTERLEAVE_FACTOR, ("io", "ii"), tail="cut")
    simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("ii"), VEC_W, ("im", "ii"), perfect=True)
        
    simple_stride_1 = stage_expr(simple_stride_1, "alpha", "alphaReg")
    simple_stride_1 = expand_dim(simple_stride_1, "alphaReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "alphaReg: _", 2)
    simple_stride_1 = hoist_const_loop(simple_stride_1, "alpha")
    simple_stride_1 = autofission(simple_stride_1, simple_stride_1.find(loop_fragment("im")).after())
        
    simple_stride_1 = stage_expr(simple_stride_1, "x[_]", "xReg")
    y_lower_bound = f"{VEC_W * INTERLEAVE_FACTOR} * io + {VEC_W} * im"
    simple_stride_1 = simplify(stage_mem(simple_stride_1, loop_fragment("ii", 2), 
                                         f"y[{y_lower_bound}:{y_lower_bound} + {VEC_W}]", "yReg"))
    
    simple_stride_1 = expand_dim(simple_stride_1, "xReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _")
    simple_stride_1 = expand_dim(simple_stride_1, "yReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "yReg : _")
    
    for buffer in ["xReg", "alphaReg", "yReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for im in _:_")
        return proc
    
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    return simplify(simple_stride_1)

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_saxpy_stride_any = specialize_precision("f32")
exo_saxpy_stride_any = rename(exo_saxpy_stride_any, exo_saxpy_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     ]
if None not in f32_instructions:
    exo_saxpy_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width, 2, C.Machine.mem_type, f32_instructions, "f32")
else:
    exo_saxpy_stride_1 = specialize_precision("f32")
    exo_saxpy_stride_1 = rename(exo_saxpy_stride_1, exo_saxpy_stride_1.name() + "_stride_1")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_daxpy_stride_any = specialize_precision("f64")
exo_daxpy_stride_any = rename(exo_daxpy_stride_any, exo_daxpy_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     ]

if None not in f64_instructions:
    exo_daxpy_stride_1 = schedule_interleave_axpy_stride_1(C.Machine.vec_width // 2, 2, C.Machine.mem_type, f64_instructions, "f64")
else:
    exo_daxpy_stride_1 = specialize_precision("f64")
    exo_daxpy_stride_1 = rename(exo_daxpy_stride_1, exo_daxpy_stride_1.name() + "_stride_1")

entry_points = [exo_saxpy_stride_any, exo_saxpy_stride_1, exo_daxpy_stride_any, exo_daxpy_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
