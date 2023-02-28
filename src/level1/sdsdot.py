from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def sdsdot_template(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    d_result: f64
    d_result = sb
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_result += d_x * d_y
    result = d_result
    
def schedule_sdsdot_stride_1(VEC_W, memory, instructions):
    pass

# TODO: add double precision instructions!
# sdsdot_stride_1 = schedule_sdsdot_stride_1(4, AVX2, [])

@proc 
def exo_sdsdot(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    sdsdot_template(n, sb, x, y, result)

"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    sdsdot_stride_1(n, sb, x, y, result)
else:
    TODO: do packing first on sub-ranges of x, then use sdsdot_stride_1 as a micro-kernel
    sdsdot_template(n, sb, x, y, result)
"""

if __name__ == "__main__":
    # print(sdsdot_stride_1)
    print(exo_sdsdot)

__all__ = ["exo_sdsdot"]