from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def dot_template(n: size, x: [f32][n], y: [f32][n], result: f32):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]
    
def schedule_dot_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(dot_template, dot_template.name() + "_simple_stride_1")
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
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simplify(simple_stride_1)

avx2_instructions = [mm256_loadu_ps, mm256_storeu_ps, avx2_select_ps,
                avx2_assoc_reduce_add_ps, mm256_setzero, mm256_fmadd_ps]
dot_stride_1 = schedule_dot_stride_1(8, AVX2, avx2_instructions)

@proc 
def dot(n: size, x: [f32][n], y: [f32][n], result: f32):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    dot_stride_1(n, x, y, result)
"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    dot_stride_1(n, x, y)
else:
    TODO: do packing first on sub-ranges of x, then use dot_stride_1 as a micro-kernel
    dot_template(n, x, y)
"""

if __name__ == "__main__":
    print(dot_stride_1)
    print(dot)

__all__ = ["dot"]