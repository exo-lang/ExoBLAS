from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def swap_template(n: size, x: [f32][n], y: [f32][n]):
    for i in seq(0, n):
        tmp: f32
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp

def schedule_swap_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(swap_template, swap_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    def stage(proc, expr_cursors, reg, cse=False):
        proc = bind_expr(proc, expr_cursors, f"{reg}", cse=cse)
        proc = expand_dim(proc, f"{reg}", VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc

    simple_stride_1 = expand_dim(simple_stride_1, "tmp", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "tmp : _")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("tmp[_] = _").after())
    
    simple_stride_1 = stage(simple_stride_1, [simple_stride_1.find("y[_]")], "yReg")
    
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("x[_] = yReg[_]").after())
    
    for buffer in ["yReg", "tmp"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
        
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    # TODO: remove once set_memory takes allocation cursor
    tail_loop_block = simple_stride_1.find("tmp = x[_]").expand(2)
    simple_stride_1 = stage_mem(simple_stride_1, tail_loop_block, "tmp", "Tmp")
    tmp_buffer = simple_stride_1.find("Tmp : _")
    xReg_buffer = tmp_buffer.prev()
    simple_stride_1 = reuse_buffer(simple_stride_1, tmp_buffer, xReg_buffer)
    simple_stride_1 = set_memory(simple_stride_1, "Tmp", DRAM)
    
    return simple_stride_1

instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32]

swap_stride_1 = schedule_swap_stride_1(C.Machine.vec_width, C.Machine.mem_type, instructions)

@proc
def exo_sswap(n: size, x: [f32][n], y: [f32][n]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    swap_stride_1(n, x, y)

"""
TODO: Should be:
if stride(x, 0) == 1:
    swap_stride_1(n, x, y)
else:
    TODO: do packing first on sub-ranges of x, then use swap_stride_1 as a micro-kernel
    swap_stride_1(n, x, y)
"""

if __name__ == "__main__":
    print(swap_stride_1)
    print(exo_sswap)

__all__ = ["exo_sswap"]