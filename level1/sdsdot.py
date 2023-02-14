from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def sdsdot_template(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    result = sb
    for i in seq(0, n):
        result += x[i] * y[i]
    
def schedule_sdsdot_stride_1(VEC_W, memory, instructions):
    # This schedule will not try to load/store as many floats from memory
    # as possible i.e. on avx2, it will load 4 floats, cast them into doubles
    # then write the the 4 floats back. We might want to write it such that
    # we load/store 8 floats at the same time, cast each half of the vector
    # into doubles, compute on it, and then write back the 8 floats together.
    simple_stride_1 = rename(sdsdot_template, sdsdot_template.name() + "_simple_stride_1")
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
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f64")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simplify(simple_stride_1)

# TODO: add double precision instructions!
sdsdot_stride_1 = schedule_sdsdot_stride_1(4, AVX2, [])

@proc 
def sdsdot(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    sdsdot_stride_1(n, sb, x, y, result)
"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    sdsdot_stride_1(n, sb, x, y, result)
else:
    TODO: do packing first on sub-ranges of x, then use sdsdot_stride_1 as a micro-kernel
    sdsdot_template(n, sb, x, y, result)
"""

if __name__ == "__main__":
    print(sdsdot_stride_1)
    print(sdsdot)

__all__ = ["sdsdot"]