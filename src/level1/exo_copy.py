from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


@proc
def scopy_template(n: size, x: [f32][n], y: [f32][n]):
    for i in seq(0, n):
        y[i] = x[i]


simple_stride_1 = rename(scopy_template, scopy_template.name() + "_simple_stride_1")
simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")


def loop_fragment(it, idx=0):
    return f"for {it} in _: _ #{idx}"


simple_stride_1 = divide_loop(
    simple_stride_1, loop_fragment("i"), C.Machine.vec_width, ("io", "ii"), tail="cut"
)
simple_stride_1 = divide_loop(
    simple_stride_1, loop_fragment("io"), 2, ("io", "im"), tail="cut"
)
simple_stride_1 = simplify(stage_mem(
    simple_stride_1,
    loop_fragment("ii"),
    f"x[(2*io+im) * {C.Machine.vec_width}:(2*io+im+1) * {C.Machine.vec_width}]",
    "xRegs",
))

simple_stride_1 = expand_dim(simple_stride_1, "xRegs", 2, "im")
simple_stride_1 = set_memory(simple_stride_1, "xRegs", C.Machine.mem_type)
simple_stride_1 = lift_alloc(simple_stride_1, "xRegs")
simple_stride_1 = replace_all(
    simple_stride_1, [C.Machine.load_instr_f32, C.Machine.store_instr_f32]
)
simple_stride_1 = fission(simple_stride_1, simple_stride_1.find(C.Machine.load_instr_f32_str).after())

scopy_stride_1 = simplify(simple_stride_1)

print(scopy_stride_1)

@proc
def exo_scopy(n: size, x: [f32][n], y: [f32][n]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    scopy_stride_1(n, x, y)


"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    scopy_stride_1(n, x, y)
else:
    TODO: do packing first on sub-ranges of x, then use scopy_stride_1 as a micro-kernel
    scopy_template(n, x, y)
"""

if __name__ == "__main__":
    print(scopy_stride_1)
    print(exo_scopy)

__all__ = ["exo_scopy"]
