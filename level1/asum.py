from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def asum_template(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])

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
    
    simple_stride_1 = bind_expr(simple_stride_1, [simple_stride_1.find("select(_)").args()[0]], "zero")
    simple_stride_1 = expand_dim(simple_stride_1, "zero", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "zero : _", n_lifts=2)
    simple_stride_1 = autofission(simple_stride_1, simple_stride_1.find("zero[_] = _").after(), n_lifts=2)
    
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_ #2", f"x[{VEC_W} * io : {VEC_W} * (io + 1)]", "xReg"))

    def stage(proc, buffer, reg):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc
    
    simple_stride_1 = stage(simple_stride_1, "xReg[_]", "xReg1")
    simple_stride_1 = stage(simple_stride_1, "-xReg[_]", "xNegReg")
    simple_stride_1 = stage(simple_stride_1, "select(_)", "selectReg")
    
    for buffer in ["xReg", "xReg1", "xNegReg", "selectReg", "zero", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    simple_stride_1 = bind_expr(simple_stride_1, "x[_] #1", "tmp")
    simple_stride_1 = bind_expr(simple_stride_1, "x[_] #2", "tmp1")
    simple_stride_1 = bind_expr(simple_stride_1, "x[_] #3", "tmp1")
    
    return simple_stride_1

"""
TODO: Add missing instructions:
- vector reg to reg copy
- vector negation
- vector reduction
"""
avx2_instructions = [mm256_loadu_ps, mm256_storeu_ps, avx2_select_ps,
                avx2_assoc_reduce_add_ps, mm256_setzero, avx2_fmadd_memu_ps]

asum_stride_1 = schedule_asum_stride_1(8, AVX2, avx2_instructions)

@proc
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    assert stride(x, 0) == 1
    asum_stride_1(n, x, result)
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
    print(asum_template)
    print(asum)
    pass

__all__ = ["asum"]
