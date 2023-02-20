from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def rotm_template_flag_neg_one(n: size, x: [f32][n], y: [f32][n], H: f32[2, 2]):
    for i in seq(0, n):
        x[i] = H[0, 0] * x[i] + H[0, 1] * y[i]
        y[i] = H[1, 0] * x[i] + H[1, 1] * y[i]

@proc
def rotm_template_flag_zero(n: size, x: [f32][n], y: [f32][n], H: f32[2, 2]):
    for i in seq(0, n):
        x[i] = x[i] + H[0, 1] * y[i]
        y[i] = H[1, 0] * x[i] + y[i]
        
@proc
def rotm_template_flag_one(n: size, x: [f32][n], y: [f32][n], H: f32[2, 2]):
    for i in seq(0, n):
        x[i] = H[0, 0] * x[i] + y[i]
        y[i] = -x[i] + H[1, 1] * y[i]

@proc
def rotm_template_flag_neg_two(n: size, x: [f32][n], y: [f32][n], H: f32[2, 2]):
    for i in seq(0, n):
        x[i] = x[i]
        y[i] = y[i]

@proc
def rotm_template(n: size, x: [f32][n], y: [f32][n], Hflag: size, H: f32[2, 2]):
    if Hflag == -1:
        rotm_template_flag_neg_one(n, x, y, H)
    if Hflag == 0:
        rotm_template_flag_zero(n, x, y, H)
    if Hflag == 1:
        rotm_template_flag_one(n, x, y, H)
    if Hflag == -2:
        rotm_template_flag_neg_two(n, x, y, H)

def schedule_rotm_stride_1(template, flag, VEC_W, memory, instructions):
    simple_stride_1 = rename(template, template.name() + "_simple_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    # Stage memories
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_", f"x[{VEC_W} * io : {VEC_W} * (io + 1)]", "xReg"))
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _", n_lifts=1)
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for ii in _:_", f"y[{VEC_W} * io : {VEC_W} * (io + 1)]", "yReg"))
    simple_stride_1 = lift_alloc(simple_stride_1, "yReg : _", n_lifts=1)
    
    def stage(proc, expr_cursors, reg, cse=False):
        proc = bind_expr(proc, expr_cursors, f"{reg}", cse=cse)
        proc = expand_dim(proc, f"{reg}", VEC_W, "ii")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc

    # No common expressions after this point, fission compute loop
    compute_loop = simple_stride_1.find("for ii in _:_")
    simple_stride_1 = fission(simple_stride_1, compute_loop.body()[0].after())
    
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
    
    # Stage constants
    if flag != 0:
        H00_cursor = simple_stride_1.find("H[0, 0]")
        simple_stride_1 = stage(simple_stride_1, [H00_cursor], "H00Reg")
        simple_stride_1 = hoist_const_loop(simple_stride_1, "H[0, 0]")
    if flag != 1:
        H01_cursor = simple_stride_1.find("H[0, 1]")
        simple_stride_1 = stage(simple_stride_1, [H01_cursor], "H01Reg")
        simple_stride_1 = hoist_const_loop(simple_stride_1, "H[0, 1]")
    if flag != 1:
        H10_cursor = simple_stride_1.find("H[1, 0]")
        simple_stride_1 = stage(simple_stride_1, [H10_cursor], "H10Reg")
        simple_stride_1 = hoist_const_loop(simple_stride_1, "H[1, 0]")
    if flag != 0:
        H11_cursor = simple_stride_1.find("H[1, 1]")
        simple_stride_1 = stage(simple_stride_1, [H11_cursor], "H11Reg")
        simple_stride_1 = hoist_const_loop(simple_stride_1, "H[1, 1]")
    
    if flag == 1:
        neg_x_cursor = simple_stride_1.find("-xReg[_]")
        simple_stride_1 = stage(simple_stride_1, [neg_x_cursor], "negXReg")
    
    # Stage binary expressions
    if flag != 0:
        x_compute_stmt = simple_stride_1.find("xReg[_] = _ + _")
        simple_stride_1 = stage(simple_stride_1, [x_compute_stmt.rhs().lhs()], "H00_Mul_X_Reg")
    if flag != 1:
        x_compute_stmt = simple_stride_1.find("xReg[_] = _ + _")
        simple_stride_1 = stage(simple_stride_1, [x_compute_stmt.rhs().rhs()], "H01_Mul_Y_Reg")
    if flag != 1:
        y_compute_stmt = simple_stride_1.find("yReg[_] = _ + _")
        simple_stride_1 = stage(simple_stride_1, [y_compute_stmt.rhs().lhs()], "H10_Mul_X_Reg")
    if flag != 0:
        y_compute_stmt = simple_stride_1.find("yReg[_] = _ + _")
        simple_stride_1 = stage(simple_stride_1, [y_compute_stmt.rhs().rhs()], "H11_Mul_Y_Reg")
    
    # Un-alias instructions
    if flag == 0:
        x_compute_stmt = simple_stride_1.find("xReg[_] = xReg[_] + _")
        simple_stride_1 = stage(simple_stride_1, [x_compute_stmt.rhs().lhs()], "xReg1")
        
        y_compute_stmt = simple_stride_1.find("yReg[_] = _ + yReg[_]")
        simple_stride_1 = stage(simple_stride_1, [y_compute_stmt.rhs().rhs()], "yReg1")
    
    for buffer in ["xReg", "xReg1", "negXReg", "yReg", "yReg1", "H00Reg", "H01Reg", "H10Reg", "H11Reg", "H00_Mul_X_Reg", "H01_Mul_Y_Reg", "H10_Mul_X_Reg", "H11_Mul_Y_Reg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    return simple_stride_1

instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32,
                C.Machine.mul_instr_f32, C.Machine.add_instr_f32,
                C.Machine.broadcast_instr_f32, C.Machine.reg_copy_instr_f32,
                C.Machine.sign_instr_f32,
                ]

if None not in instructions:
    rotm_flag_neg_one_stride_1 = schedule_rotm_stride_1(rotm_template_flag_neg_one, -1, C.Machine.vec_width, C.Machine.mem_type, instructions)
else:
    rotm_flag_neg_one_stride_1 = rotm_template_flag_neg_one

if None not in instructions:
    rotm_flag_zero_stride_1 = schedule_rotm_stride_1(rotm_template_flag_zero, 0, C.Machine.vec_width, C.Machine.mem_type, instructions)
else:
    rotm_flag_zero_stride_1 = rotm_template_flag_zero

if None not in instructions:
    rotm_flag_one_stride_1 = schedule_rotm_stride_1(rotm_template_flag_one, 1, C.Machine.vec_width, C.Machine.mem_type, instructions)
else:
    rotm_flag_one_stride_1 = rotm_template_flag_one

@proc
def rotm_template_stride_1(n: size, x: [f32][n], y: [f32][n], Hflag: size, H: f32[2, 2]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    if Hflag == -1:
        rotm_flag_neg_one_stride_1(n, x, y, H)
    if Hflag == 0:
        rotm_flag_zero_stride_1(n, x, y, H)
    if Hflag == 1:
        rotm_flag_one_stride_1(n, x, y, H)
    if Hflag == -2:
        rotm_template_flag_neg_two(n, x, y, H)

@proc
def exo_srotm(n: size, x: [f32][n], y: [f32][n], Hflag: size, H: f32[2, 2]):
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    rotm_template_stride_1(n, x, y, Hflag, H)

"""
TODO: Should be:
if stride(x, 0) == 1 and stride(y, 0) == 1:
    rot_stride_1(n, x, y, c, s)
else:
    TODO: do packing first on sub-ranges of x, then use rot_stride_1 as a micro-kernel
    rot_template(n, x, y, c, s)
"""

if __name__ == "__main__":
    print(rotm_flag_neg_one_stride_1)
    print(exo_srotm)

__all__ = ["exo_srotm"]