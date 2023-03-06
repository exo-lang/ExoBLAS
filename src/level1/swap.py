from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def swap_template(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        tmp: R
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp

def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(swap_template, "exo_" + prefix + "swap")
    for arg in ["x", "y", "tmp"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy

def schedule_swap_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
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
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
        
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    # TODO: remove once set_memory takes allocation cursor
    tail_loop_block = simple_stride_1.find("tmp = x[_]").expand(2)
    simple_stride_1 = stage_mem(simple_stride_1, tail_loop_block, "tmp", "Tmp")
    tmp_buffer = simple_stride_1.find("Tmp : _")
    xReg_buffer = tmp_buffer.prev()
    simple_stride_1 = reuse_buffer(simple_stride_1, tmp_buffer, xReg_buffer)
    simple_stride_1 = set_memory(simple_stride_1, "Tmp", DRAM)
    
    return simple_stride_1

exo_sswap_stride_any = specialize_precision("f32")
exo_sswap_stride_any = rename(exo_sswap_stride_any, exo_sswap_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32]
exo_sswap_stride_1 = schedule_swap_stride_1(C.Machine.vec_width, C.Machine.mem_type, f32_instructions, "f32")

exo_dswap_stride_any = specialize_precision("f64")
exo_dswap_stride_any = rename(exo_dswap_stride_any, exo_dswap_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64, C.Machine.store_instr_f64]
exo_dswap_stride_1 = schedule_swap_stride_1(C.Machine.vec_width // 2, C.Machine.mem_type, f64_instructions, "f64")

entry_points = [exo_sswap_stride_any, exo_sswap_stride_1, exo_dswap_stride_any, exo_dswap_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]