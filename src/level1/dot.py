from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def sdot_template(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]

def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(sdot_template, "exo_" + prefix + "dot")
    for arg in ["x", "y", "result"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy

def schedule_dot_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True))
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").before())
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").after())
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")
    
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{VEC_W} * io: {VEC_W} * (io + 1)]", "xReg")
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"y[{VEC_W} * io: {VEC_W} * (io + 1)]", "yReg")
    
    for buffer in ["xReg", "yReg", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simplify(simple_stride_1)

def schedule_dot_stride_1_interleaved(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W * INTERLEAVE_FACTOR, ("io", "ii"), tail = "cut")
    simple_stride_1 = divide_loop(simple_stride_1, "for ii in _:_", VEC_W, ("im", "ii"), perfect=True)
    
    simple_stride_1 = reorder_loops(simple_stride_1, "io im")
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True))
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _", n_lifts=2)
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").before(), n_lifts=2)
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").after(), n_lifts=2)
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")
    simple_stride_1 = reorder_loops(simple_stride_1, "im io")

    lower_bound = f"{VEC_W * INTERLEAVE_FACTOR} * io + im * {VEC_W}"
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg")
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"y[{lower_bound}: {lower_bound} + {VEC_W}]", "yReg")
    
    for buffer in ["xReg", "yReg", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    simple_stride_1 = simplify(simple_stride_1)
    
    simple_stride_1 = expand_dim(simple_stride_1, "xReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _")
    simple_stride_1 = expand_dim(simple_stride_1, "yReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "yReg : _")
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
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    
    return simplify(simple_stride_1)

def schedule_dot_stride_any(VEC_W, incX, incY, precision, stride_suffix):
    stride_any = specialize_precision(precision)
    stride_any = rename(stride_any, stride_any.name() + f"_stride_{stride_suffix}")
    if incX != None:
        stride_any = stride_any.add_assertion(f"stride(x, 0) == {incX}")
    if incY != None:
        stride_any = stride_any.add_assertion(f"stride(y, 0) == {incY}")

    if VEC_W is not None:
        stride_any = divide_loop(stride_any, "for i in _:_", VEC_W, ("io", "ii"), tail="cut")
        
        stride_any = reorder_loops(stride_any, "io ii")
        stride_any = simplify(stage_mem(stride_any, "for io in _:_", "result", "resultReg", accum=True))
        stride_any = expand_dim(stride_any, "resultReg :_ ", VEC_W, "ii")
        stride_any = lift_alloc(stride_any, "resultReg : _", n_lifts=1)
        stride_any = fission(stride_any, stride_any.find("for io in _:_").before(), n_lifts=1)
        stride_any = fission(stride_any, stride_any.find("for io in _:_").after(), n_lifts=1)
        stride_any = reorder_loops(stride_any, "ii io")
        stride_any = set_memory(stride_any, "resultReg", DRAM_STATIC)
        
        stride_any = unroll_loop(stride_any, "for ii in _:_")
        stride_any = unroll_loop(stride_any, "for ii in _:_")
        stride_any = unroll_loop(stride_any, "for ii in _:_")
        
        for i in range(VEC_W):
            stride_any = bind_expr(stride_any, f"x[{i} + {VEC_W} * io]", f"x{i}")
            stride_any = set_precision(stride_any, f"x{i}", precision)
            for j in range(i * 2):
                stride_any = reorder_stmts(stride_any, stride_any.find(f"x{i} : _").expand(-1))
            for j in range(i):
                stride_any = reorder_stmts(stride_any, stride_any.find(f"x{i} = _").expand(-1))
        for i in range(VEC_W):
            stride_any = bind_expr(stride_any, f"y[{i} + {VEC_W} * io]", f"y{i}")
            stride_any = set_precision(stride_any, f"y{i}", precision)
            for j in range(i * 2):
                stride_any = reorder_stmts(stride_any, stride_any.find(f"y{i} : _").expand(-1))
            for j in range(i):
                stride_any = reorder_stmts(stride_any, stride_any.find(f"y{i} = _").expand(-1))
    
    return stride_any

#################################################
# Kernel Parameters
#################################################

STRIDE_1_INTERLEAVE_FACTOR = 4
STRIDE_ANY_VECTORIZATION_FACTOR_F32 = C.Machine.vec_width // 2
STRIDE_ANY_VECTORIZATION_FACTOR_F64 = STRIDE_ANY_VECTORIZATION_FACTOR_F32 // 2

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sdot_stride_any = schedule_dot_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F32, None, None, "f32", "any")

f32_instructions = [C.Machine.load_instr_f32, 
                     C.Machine.store_instr_f32,
                     C.Machine.assoc_reduce_add_instr_f32,
                     C.Machine.set_zero_instr_f32,
                     C.Machine.fmadd_instr_f32]

if None not in f32_instructions:
    exo_sdot_stride_1 = schedule_dot_stride_1_interleaved(C.Machine.vec_width, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32")
else:
    exo_sdot_stride_1 = schedule_dot_stride_any(C.Machine.vec_width, 1, 1, "f32", "1")
    
#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_ddot_stride_any = schedule_dot_stride_any(STRIDE_ANY_VECTORIZATION_FACTOR_F64, None, None, "f64", "any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.assoc_reduce_add_instr_f64,
                     C.Machine.set_zero_instr_f64,
                     C.Machine.fmadd_instr_f64
                     ]

if None not in f64_instructions:
    exo_ddot_stride_1 = schedule_dot_stride_1_interleaved(C.Machine.vec_width // 2, STRIDE_1_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64")
else:
    exo_ddot_stride_1 = schedule_dot_stride_any(C.Machine.vec_width // 2, 1, 1, "f64", "any")

entry_points = [exo_sdot_stride_any, exo_sdot_stride_1, exo_ddot_stride_any, exo_ddot_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]