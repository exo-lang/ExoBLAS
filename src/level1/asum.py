from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def asum_template(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])
        
@instr("{dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));")
def avx2_abs_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = select(0.0, src[i], src[i], -src[i])

def schedule_asum_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(asum_template, asum_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
            
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True))
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").before())
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").after())
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")

    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{VEC_W} * io : {VEC_W} * (io + 1)]", "xReg"))

    def stage(proc, buffer, reg):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=1)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc
    
    simple_stride_1 = stage(simple_stride_1, "select(_)", "selectReg")
    
    for buffer in ["xReg", "selectReg", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    for i in range(4):
        select_cursor = simple_stride_1.find("select(_, _, _, _)")
        arg_cursor = select_cursor.args()[i]
        simple_stride_1 = bind_expr(simple_stride_1, [arg_cursor], f"tmp_{i}")
        
    return simple_stride_1

def schedule_asum_stride_1_interleaved(VEC_W, INTERLEAVE_FACTOR, memory, instructions):
    simple_stride_1 = rename(asum_template, asum_template.name() + "_simple_stride_1_interleaved")
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
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg"))

    def stage(proc, buffer, reg):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=1)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc
    
    simple_stride_1 = stage(simple_stride_1, "select(_)", "selectReg")
    
    for buffer in ["xReg", "selectReg", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    for i in range(4):
        select_cursor = simple_stride_1.find("select(_, _, _, _)")
        arg_cursor = select_cursor.args()[i]
        simple_stride_1 = bind_expr(simple_stride_1, [arg_cursor], f"tmp_{i}")

    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for im in _:_")
        return proc
        
    simple_stride_1 = expand_dim(simple_stride_1, "xReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _")
    simple_stride_1 = expand_dim(simple_stride_1, "selectReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "selectReg : _")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    
    return simple_stride_1

instructions = [C.Machine.load_instr_f32,
                C.Machine.store_instr_f32, 
                C.Machine.assoc_reduce_add_instr_f32,
                C.Machine.set_zero_instr_f32,
                C.Machine.reduce_add_wide_instr_f32,
                avx2_abs_ps if C.Machine.mem_type is AVX2 else None]

INTERLEAVE_FACTOR = 2

if None not in instructions:
    asum_stride_1 = schedule_asum_stride_1_interleaved(C.Machine.vec_width, INTERLEAVE_FACTOR, C.Machine.mem_type, instructions)
    asum_stride_1_small_n = schedule_asum_stride_1(C.Machine.vec_width, C.Machine.mem_type, instructions)
else:
    asum_stride_1 = asum_template
    for i in range(4):
        select_cursor = asum_stride_1.find("select(_, _, _, _)")
        arg_cursor = select_cursor.args()[i]
        asum_stride_1 = bind_expr(asum_stride_1, [arg_cursor], f"tmp_{i}")
    asum_stride_1_small_n = asum_stride_1

small_threshold = INTERLEAVE_FACTOR * C.Machine.vec_width

@proc
def exo_sasum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    assert stride(x, 0) == 1
    if n >= small_threshold:
        asum_stride_1(n, x, result)
    else:
        asum_stride_1_small_n(n, x, result)

"""
TODO: Should be:
if stride(x, 0) == 1:
    asum_stride_1(n, x, result)
else:
    TODO: do packing first on sub-ranges of x, then use asum_stride_1 as a micro-kernel
    asum_template(n, x, result)
"""

if __name__ == "__main__":
    print(asum_stride_1)
    print(asum_stride_1_small_n)
    print(asum_template)
    print(exo_sasum)
    pass

__all__ = ["exo_sasum"]
