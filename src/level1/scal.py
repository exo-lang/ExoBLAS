from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def scal_template(n: size, alpha: f32, x: [f32][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]

def schedule_scal_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(scal_template, scal_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    def stage(proc, expr_cursors, reg, cse=False):
        proc = bind_expr(proc, expr_cursors, f"{reg}", cse=cse)
        proc = expand_dim(proc, f"{reg}", VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc

    simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("x[_]")], "xReg")
    simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("alpha")], "alphaReg")
    simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("alphaReg[_] * xReg[_]")], "mulReg")
    
    for buffer in ["xReg", "alphaReg", "mulReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
        
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simple_stride_1

instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32,
                C.Machine.mul_instr_f32, C.Machine.broadcast_scalar_instr_f32,
                ]

scal_stride_1 = schedule_scal_stride_1(C.Machine.vec_width, C.Machine.mem_type, instructions)

@proc
def exo_sscal(n: size, alpha: f32, x: [f32][n]):
    assert stride(x, 0) == 1
    scal_stride_1(n, alpha, x)

"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    rot_stride_1(n, x, y, c, s)
else:
    TODO: do packing first on sub-ranges of x, then use rot_stride_1 as a micro-kernel
    rot_template(n, x, y, c, s)
"""

if __name__ == "__main__":
    print(scal_stride_1)
    print(exo_sscal)

__all__ = ["exo_sscal"]