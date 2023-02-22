from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def axpy_template(
  n: size,
  alpha: f32,
  x: [f32][n],
  y: [f32][n],
):
    for i in seq(0, n):
        y[i] += alpha * x[i]

def schedule_axpy_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(axpy_template, axpy_template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    def stage_expr(proc, expr, buffer):
        proc = bind_expr(proc, expr, buffer)
        proc = expand_dim(proc, buffer, VEC_W, "ii")
        proc = lift_alloc(proc, f"{buffer}:_", n_lifts=2)
        proc = fission(proc, proc.find(f"{buffer}[_] = _").after())
        return proc
    
    def hoist_const_loop(proc, constant):
        while True:
            try:
                loop_cursor = proc.find(constant).parent().parent()
                proc = reorder_stmts(proc, loop_cursor.expand(-1))
            except:
                break
        loop_cursor = proc.find(constant).parent().parent()
        proc = autofission(proc, loop_cursor.after())
        return proc

    loop_fragment = lambda iter, idx=0: f"for {iter} in _:_ #{idx}"
    simple_stride_1 = divide_loop(simple_stride_1, loop_fragment("i"), VEC_W, ("io", "ii"), tail="cut")

    simple_stride_1 = stage_expr(simple_stride_1, "alpha", "alphaReg")
    simple_stride_1 = stage_expr(simple_stride_1, "x[_]", "xReg")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, loop_fragment("ii", 2), f"y[io * {VEC_W}:(io + 1) * {VEC_W}]", "yReg"))
    
    simple_stride_1 = hoist_const_loop(simple_stride_1, "alpha")
    
    for buffer in ["xReg", "alphaReg", "yReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simplify(simple_stride_1)

instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     ]

if None not in  instructions:
    axpy_stride_1 = schedule_axpy_stride_1(C.Machine.vec_width, C.Machine.mem_type, instructions)
else:
    axpy_stride_1 = axpy_template

@proc
def exo_saxpy(n: size, alpha: f32, x: [f32][n], y: [f32][n]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    axpy_stride_1(n, alpha, x, y)
"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    axpy_stride_1(n, alpha, x, y)
else:
    TODO: do packing first on sub-ranges of x, then use axpy_stride_1 as a micro-kernel
    axpy_template(n, alpha, x, y)
"""

if __name__ == "__main__":
    print(axpy_stride_1)
    print(exo_saxpy)

__all__ = ["exo_saxpy"]
