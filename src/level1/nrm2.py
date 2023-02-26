from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def nrm2_template(n: size, x: [f32][n], result: f32):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * x[i]
        
def schedule_nrm2_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(nrm2_template, nrm2_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True))
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").before())
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").after())
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")
    
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{VEC_W} * io: {VEC_W} * (io + 1)]", "xReg")
    simple_stride_1 = bind_expr(simple_stride_1, "xReg[_] #1", "xReg1")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _", n_lifts=2)
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("xReg1[_] = _").after())
    simple_stride_1 = simplify(simple_stride_1)
    
    for buffer in ["xReg", "xReg1", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simple_stride_1

def schedule_nrm2_stride_1_interleaved(VEC_W, INTERLEAVE_FACTOR, memory, instructions):
    simple_stride_1 = rename(nrm2_template, nrm2_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    
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
    
    lower_bound = f"{VEC_W} * im + {VEC_W * INTERLEAVE_FACTOR} * io"
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg")
    simple_stride_1 = bind_expr(simple_stride_1, "xReg[_] #1", "xReg1")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _", n_lifts=1)
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("xReg1[_] = _").after())
    simple_stride_1 = simplify(simple_stride_1)
    
    for buffer in ["xReg", "xReg1", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    simple_stride_1 = expand_dim(simple_stride_1, "xReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _")
    
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
    
    return simple_stride_1

instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32, 
                     C.Machine.set_zero_instr_f32, C.Machine.fmadd_instr_f32,
                     C.Machine.reg_copy_instr_f32, C.Machine.assoc_reduce_add_instr_f32,
                     ]

if None not in instructions:
    nrm2_stride_1 = schedule_nrm2_stride_1_interleaved(C.Machine.vec_width, 2, C.Machine.mem_type, instructions)
else:
    nrm2_stride_1 = nrm2_template

print(nrm2_stride_1)

# TODO: this calculates ||x||^2, not ||x|| 
@proc 
def exo_snrm2(n: size, x: [f32][n], result: f32):
    assert stride(x, 0) == 1
    nrm2_stride_1(n, x, result)
    
"""
TODO: Should be:
if stride(x, 0) == 1:
    nrm2_stride_1(n, x, result)
else:
    TODO: do packing first on sub-ranges of x, then use nrm2_stride_1 as a micro-kernel
    nrm2_template(n, x, result)
"""

if __name__ == "__main__":
    print(nrm2_stride_1)
    print(exo_snrm2)

__all__ = ["exo_snrm2"]