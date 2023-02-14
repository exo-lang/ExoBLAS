from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


@proc
def copy_template(n: size, x: f32[n], y: f32[n]):
    for i in seq(0, n):
        y[i] = x[i]


simple_stride_1 = rename(copy_template, copy_template.name() + "_simple_stride_1")
simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")


def loop_fragment(it, idx=0):
    return f"for {it} in _: _ #{idx}"


simple_stride_1 = divide_loop(
    simple_stride_1, loop_fragment("i"), C.Machine.vec_width, ("io", "ii"), tail="cut"
)
simple_stride_1 = stage_mem(
    simple_stride_1,
    loop_fragment("ii"),
    f"x[io * {C.Machine.vec_width}:(io + 1) * {C.Machine.vec_width}]",
    "xReg",
)

simple_stride_1 = set_memory(simple_stride_1, "xReg", C.Machine.mem_type)
simple_stride_1 = replace_all(
    simple_stride_1, [C.Machine.load_instr, C.Machine.store_instr]
)

copy_stride_1 = simplify(simple_stride_1)


@proc
def exo_bcopy(n: size, x: f32[n], y: f32[n]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    copy_stride_1(n, x, y)


"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    copy_stride_1(n, x, y)
else:
    TODO: do packing first on sub-ranges of x, then use asum_stride_1 as a micro-kernel
    copy_template(n, x, y)
"""

if __name__ == "__main__":
    print(copy_stride_1)
    print(exo_bcopy)

__all__ = ["exo_bcopy"]
