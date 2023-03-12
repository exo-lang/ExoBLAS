from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def rot_template(n: size, x: [R][n], y: [R][n], c: R, s: R):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = c * xReg + s * y[i]
        y[i] = -s * xReg + c * y[i]

def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(rot_template, "exo_" + prefix + "rot")
    for arg in ["x", "y", "c", "s", "xReg"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy

def schedule_rot_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    # Stage memories
    simple_stride_1 = expand_dim(simple_stride_1, "xReg", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _", n_lifts=2)
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("xReg[_] = _").after())
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_ #1", f"y[{VEC_W} * io : {VEC_W} * (io + 1)]", "yReg"))    
    simple_stride_1 = lift_alloc(simple_stride_1, "yReg : _", n_lifts=1)

    def stage(proc, expr_cursors, reg, cse=False):
        proc = bind_expr(proc, expr_cursors, f"{reg}", cse=cse)
        proc = expand_dim(proc, f"{reg}", VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc

    # Stage constants
    compute_loop = simple_stride_1.find("for ii in _:_ #1")
    simple_stride_1 = stage(simple_stride_1, [compute_loop.body()[0].rhs().lhs().lhs(), compute_loop.body()[1].rhs().rhs().lhs()], "cReg", cse=True)
    compute_loop = simple_stride_1.find("for ii in _:_ #2")
    simple_stride_1 = stage(simple_stride_1, [compute_loop.body()[0].rhs().rhs().lhs()], "sReg", cse=True)
    
    # No common expressions after this point, fission compute loop
    compute_loop = simple_stride_1.find("for ii in _:_ #3")
    simple_stride_1 = fission(simple_stride_1, compute_loop.body()[0].after())        
    
    # Stage -s
    compute_loop2 = simple_stride_1.find("for ii in _:_ #4")
    simple_stride_1 = stage(simple_stride_1, [compute_loop2.body()[0].rhs().lhs().lhs()], "sNegReg", cse=True)
    
    # Const loops hoisting
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

    simple_stride_1 = hoist_const_loop(simple_stride_1, "c")
    simple_stride_1 = hoist_const_loop(simple_stride_1, "s")
    simple_stride_1 = hoist_const_loop(simple_stride_1, "-s")
            
    simple_stride_1 = bind_expr(simple_stride_1, "-s", "NegS")
    simple_stride_1 = set_precision(simple_stride_1, "NegS", precision)
    simple_stride_1 = lift_alloc(simple_stride_1, "NegS")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("NegS = -s").after())
    simple_stride_1 = remove_loop(simple_stride_1, "for ii in _:_ #2")
    
    # Stage binary expressions
    compute_loop1 = simple_stride_1.find("for ii in _:_ #4")
    simple_stride_1 = stage(simple_stride_1, [compute_loop1.body()[0].rhs().lhs()], "C_Mul_X_Reg")
    compute_loop1 = simple_stride_1.find("for ii in _:_ #5")
    simple_stride_1 = stage(simple_stride_1, [compute_loop1.body()[0].rhs().rhs()], "S_Mul_Y_Reg")
    compute_loop1 = simple_stride_1.find("for ii in _:_ #6")
    simple_stride_1 = stage(simple_stride_1, [compute_loop1.body()[0].rhs()], "cX_Add_sY_Reg")
    compute_loop2 = simple_stride_1.find("for ii in _:_ #8")
    simple_stride_1 = stage(simple_stride_1, [compute_loop2.body()[0].rhs().lhs()], "SNeg_Mul_X_Reg")
    compute_loop2 = simple_stride_1.find("for ii in _:_ #9")
    simple_stride_1 = stage(simple_stride_1, [compute_loop2.body()[0].rhs().rhs()], "C_Mul_Y_Reg")
    
    for buffer in ["xReg", "yReg", "cReg", "sReg", "sNegReg", "cX_Add_sY_Reg", "C_Mul_X_Reg", "S_Mul_Y_Reg", "SNeg_Mul_X_Reg", "C_Mul_Y_Reg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    # TODO: remove once set_memory takes allocation cursor
    tail_loop_block = simple_stride_1.find("xReg = x[_]").expand(2)
    simple_stride_1 = stage_mem(simple_stride_1, tail_loop_block, "xReg", "xTmp")
    tmp_buffer = simple_stride_1.find("xTmp : _")
    xReg_buffer = tmp_buffer.prev()
    simple_stride_1 = reuse_buffer(simple_stride_1, tmp_buffer, xReg_buffer)
    simple_stride_1 = set_memory(simple_stride_1, "xTmp", DRAM)
    
    return simple_stride_1
    
#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_srot_stride_any = specialize_precision("f32")
exo_srot_stride_any = rename(exo_srot_stride_any, exo_srot_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32, 
                    C.Machine.store_instr_f32,
                    C.Machine.mul_instr_f32, 
                    C.Machine.add_instr_f32,
                    C.Machine.broadcast_scalar_instr_f32,
                    ]

if None not in f32_instructions:
    exo_srot_stride_1 = schedule_rot_stride_1(C.Machine.vec_width, C.Machine.mem_type, f32_instructions, "f32")
else:
    exo_srot_stride_1 = specialize_precision("f32")
    exo_srot_stride_1 = rename(exo_srot_stride_1, exo_srot_stride_1.name() + "_stride_1")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_drot_stride_any = specialize_precision("f64")
exo_drot_stride_any = rename(exo_drot_stride_any, exo_drot_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                    C.Machine.store_instr_f64,
                    C.Machine.mul_instr_f64, 
                    C.Machine.add_instr_f64,
                    C.Machine.broadcast_scalar_instr_f64,
                    ]

if None not in f64_instructions:
    exo_drot_stride_1 = schedule_rot_stride_1(C.Machine.vec_width // 2, C.Machine.mem_type, f64_instructions, "f64")
else:
    exo_drot_stride_1 = specialize_precision("f64")
    exo_drot_stride_1 = rename(exo_drot_stride_1, exo_drot_stride_1.name() + "_stride_1")
    
entry_points = [exo_srot_stride_any, exo_srot_stride_1, exo_drot_stride_any, exo_drot_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]