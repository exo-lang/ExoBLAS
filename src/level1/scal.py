from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def scal_template(n: size, alpha: R, x: [R][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]

@proc
def scal_template_alpha_0(n: size, x: [R][n]):
    for i in seq(0, n):
        x[i] = 0.0

def specialize_scal(precision, alpha):
    prefix = "s" if precision == "f32" else "d"
    specialized_scal = scal_template if alpha != 0 else scal_template_alpha_0
    specialized_scal_name = specialized_scal.name()
    specialized_scal_name = specialized_scal_name.replace("_template", "")
    specialized_scal = rename(specialized_scal, "exo_" + prefix + specialized_scal_name)
    
    args = ["x"]
    if alpha != 0:
        args.append("alpha")
        
    for arg in args:
        specialized_scal = set_precision(specialized_scal, arg, precision)
    return specialized_scal

def schedule_scal_stride_1(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision, alpha):
    simple_stride_1 = specialize_scal(precision, alpha)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    
    if alpha != 0:
        simple_stride_1 = stage_mem(simple_stride_1, simple_stride_1.body(), "alpha", "alpha_")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail="cut")
    
    def stage(proc, expr_cursors, reg, cse=False):
        proc = bind_expr(proc, expr_cursors, f"{reg}", cse=cse)
        proc = expand_dim(proc, f"{reg}", VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=1)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc
    
    if alpha != 0:
        constantReg = "alphaReg"
        registers = ["xReg", "alphaReg", "mulReg"]
        simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("x[_]")], "xReg")
        simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("alpha_")], "alphaReg")
        simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("alphaReg[_] * xReg[_]")], "mulReg")
    else:
        constantReg = "zeroReg"
        registers = ["zeroReg"]
        simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("0.0")], "zeroReg")
    
    for buffer in registers:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
    
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
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    if INTERLEAVE_FACTOR > 1:
        simple_stride_1 = divide_loop(simple_stride_1, "for io in _:_", INTERLEAVE_FACTOR, ("io", "im"), tail="cut")
        
        for reg in registers:
            simple_stride_1 = expand_dim(simple_stride_1, reg, INTERLEAVE_FACTOR, "im")
            simple_stride_1 = lift_alloc(simple_stride_1, reg, n_lifts=2)
            
        simple_stride_1 = lift_alloc(simple_stride_1, f"{constantReg} #1", n_lifts=1) # Tail loop to allow broadcast hoisting
        
        simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    else:
        simple_stride_1 = lift_alloc(simple_stride_1, f"{constantReg}", n_lifts=1) # Main loop to allow broadcast hoisting
    
    if alpha != 0:
        for i in range(INTERLEAVE_FACTOR + 1):
            simple_stride_1 = hoist_const_broadcast(simple_stride_1, f"alpha_ #{i}")
    else:
        zero_instr = C.Machine.set_zero_instr_f32 if precision == "f32" else C.Machine.set_zero_instr_f64
        for i in range(INTERLEAVE_FACTOR + 1):
            zero_call_cursor = simple_stride_1.find(f"{zero_instr.name()}(_) #{i}")
            simple_stride_1 = autofission(simple_stride_1, zero_call_cursor.after())

    return simple_stride_1

def schedule_scal_stride_any(VEC_W, alpha, incX, precision, stride_suffix):
    stride_any = specialize_scal(precision, alpha)
    stride_any = rename(stride_any, stride_any.name() + f"_stride_{stride_suffix}")
    if incX != None:
        stride_any = stride_any.add_assertion(f"stride(x, 0) == {incX}")
    if alpha != 0:
        stride_any = stage_mem(stride_any, stride_any.body(), "alpha", "alpha_")

    if VEC_W is not None:    
        stride_any = divide_loop(stride_any, "for i in _:_", VEC_W, ("io", "ii"), tail="cut")
        stride_any = unroll_loop(stride_any, "for ii in _:_")
        
        if alpha != 0:
            for i in range(VEC_W):
                stride_any = bind_expr(stride_any, f"x[{VEC_W} * io + {i}]", f"x{i}")
                stride_any = set_precision(stride_any, f"x{i}", precision)
                for j in range(i * 2):
                    stride_any = reorder_stmts(stride_any, stride_any.find(f"x{i} : _").expand(-1))
                for j in range(i):
                    stride_any = reorder_stmts(stride_any, stride_any.find(f"x{i} = _").expand(-1))
    
    return stride_any

#################################################
# Kernel Parameters
#################################################

STRIDE_1_INTERLEAVE_FACTOR = 4
STRIDE_2_VECTORIZATION_FACTOR_F32 = C.Machine.vec_width
STRIDE_2_VECTORIZATION_FACTOR_F64 = STRIDE_2_VECTORIZATION_FACTOR_F32 // 2
STRIDE_ANY_VECTORIZATION_FACTOR_F32 = C.Machine.vec_width // 2
STRIDE_ANY_VECTORIZATION_FACTOR_F64 = STRIDE_ANY_VECTORIZATION_FACTOR_F32 // 2

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sscal_stride_any = schedule_scal_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F32, None, None, "f32", "any")
exo_sscal_alpha_0_stride_any = schedule_scal_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F32, 0, None, "f32", "any")

f32_instructions = [C.Machine.load_instr_f32,
                    C.Machine.store_instr_f32,
                    C.Machine.mul_instr_f32, 
                    C.Machine.broadcast_scalar_instr_f32,
                    C.Machine.set_zero_instr_f32,
                    ]

if None not in f32_instructions:
    exo_sscal_stride_1 = schedule_scal_stride_1(C.Machine.vec_width, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32", None)
    exo_sscal_alpha_0_stride_1 = schedule_scal_stride_1(C.Machine.vec_width, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32", 0)
else:
    exo_sscal_stride_1 = schedule_scal_stride_any(C.Machine.vec_width, None, 1, "f32", "1")
    exo_sscal_alpha_0_stride_1 = schedule_scal_stride_any(C.Machine.vec_width, 0, 1, "f32", "1")

exo_sscal_stride_2 = schedule_scal_stride_any(STRIDE_2_VECTORIZATION_FACTOR_F32, None, 2, "f32", "2")
exo_sscal_alpha_0_stride_2 = schedule_scal_stride_any(STRIDE_2_VECTORIZATION_FACTOR_F32, 0, 2, "f32", "2")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dscal_stride_any = schedule_scal_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F64, None, None, "f64", "any")
exo_dscal_alpha_0_stride_any = schedule_scal_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F64, 0, None, "f64", "any")

f64_instructions = [C.Machine.load_instr_f64,
                    C.Machine.store_instr_f64,
                    C.Machine.mul_instr_f64, 
                    C.Machine.broadcast_scalar_instr_f64,
                    C.Machine.set_zero_instr_f64,
                    ]

if None not in f64_instructions:
    exo_dscal_stride_1 = schedule_scal_stride_1(C.Machine.vec_width // 2, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64", None)
    exo_dscal_alpha_0_stride_1 = schedule_scal_stride_1(C.Machine.vec_width // 2, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64", 0)
else:
    exo_dscal_stride_1 = schedule_scal_stride_any(4, None, 1, "f64", "1")
    exo_dscal_alpha_0_stride_1 = schedule_scal_stride_any(4, 0, 1, "f64", "1")

exo_dscal_stride_2 = schedule_scal_stride_any(STRIDE_2_VECTORIZATION_FACTOR_F64, None, 2, "f64", "2")
exo_dscal_alpha_0_stride_2 = schedule_scal_stride_any(STRIDE_2_VECTORIZATION_FACTOR_F64, 0, 2, "f64", "2")

entry_points = [exo_sscal_stride_any, exo_sscal_stride_1, exo_sscal_alpha_0_stride_1, exo_sscal_alpha_0_stride_any, exo_sscal_stride_2, exo_sscal_alpha_0_stride_2,
                exo_dscal_stride_any, exo_dscal_stride_1, exo_dscal_alpha_0_stride_1, exo_dscal_alpha_0_stride_any, exo_dscal_stride_2, exo_dscal_alpha_0_stride_2]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]