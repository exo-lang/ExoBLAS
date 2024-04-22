from __future__ import annotations

from exo.platforms.neon import *
from exo import instr
from exo.stdlib.scheduling import *

from machines.machine_params import MachineParameters
from stdlib import *


@instr("{dst_data} = {src_data};")
def neon_reg_copy_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


# TODO: add to EXO's Neon library
@instr("*{result} += vaddvq_f32({x_data});")
def neon_assoc_reduce_add_instr_4xf32(result: f32 @ DRAM, x: [f32][4] @ Neon):
    assert stride(x, 0) == 1
    for i in seq(0, 4):
        result += x[i]


@instr("{result_data} += vaddvq_f32({x_data});")
def neon_assoc_reduce_add_instr_4xf32_buffer(result: [f32][1] @ DRAM, x: [f32][4] @ Neon):
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1
    for i in seq(0, 4):
        result[0] += x[i]


@instr("{result_data} += vaddvq_f64({x_data});")
def neon_assoc_reduce_add_instr_2xf64_buffer(result: [f64][1] @ DRAM, x: [f64][2] @ Neon):
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1
    for i in seq(0, 2):
        result[0] += x[i]


@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32_backawrds(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[3 - i]


@instr("vst1q_f32(&{dst_data}, {src_data});")
def neon_vst_4xf32_backawrds(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[3 - i] = src[i]


@instr("{dst_data} = vld1q_f64(&{src_data});")
def neon_vld_2xf64_backwards(dst: [f64][2] @ Neon, src: [f64][2] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 2):
        dst[i] = src[1 - i]


@instr("vst1q_f64(&{dst_data}, {src_data});")
def neon_vst_2xf64_backwards(dst: [f64][2] @ DRAM, src: [f64][2] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 2):
        dst[1 - i] = src[i]


neon_vfmadd_reduce_4xf32_4xf32 = rename(neon_vfmadd_4xf32_4xf32, "neon_vfmadd_reduce_4xf32_4xf32")
neon_vfmadd_reduce_2xf64_2xf64 = rename(neon_vfmadd_2xf64_2xf64, "neon_vfmadd_reduce_2xf64_2xf64")


@instr("{dst_data} = vmlaq_f32({acc_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_4xf32(
    dst: [f32][4] @ Neon,
    acc: [f32][4] @ Neon,
    lhs: [f32][4] @ Neon,
    rhs: [f32][4] @ Neon,
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = acc[i] + lhs[i] * rhs[i]


@instr("{dst_data} = vmlaq_f64({acc_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_2xf64_2xf64(
    dst: [f64][2] @ Neon,
    acc: [f64][2] @ Neon,
    lhs: [f64][2] @ Neon,
    rhs: [f64][2] @ Neon,
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 2):
        dst[i] = acc[i] + lhs[i] * rhs[i]


Machine = MachineParameters(
    name="neon",
    mem_type=Neon,
    n_vec_registers=32,
    f32_vec_width=4,
    vec_units=4,
    supports_predication=False,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    instrs=[
        neon_vld_4xf32,
        neon_vld_4xf32_backawrds,
        neon_vst_4xf32,
        neon_vst_4xf32_backawrds,
        neon_broadcast_4xf32,
        neon_broadcast_4xf32_scalar,
        neon_vfmadd_4xf32_4xf32,
        neon_vfmadd_reduce_4xf32_4xf32,
        neon_zero_4xf32,
        neon_assoc_reduce_add_instr_4xf32,
        neon_vmul_4xf32,
        neon_vadd_4xf32,
        neon_reduce_vadd_4xf32,
        neon_reg_copy_4xf32,
        neon_vneg_4xf32,
        neon_assoc_reduce_add_instr_4xf32_buffer,
        neon_vld_2xf64,
        neon_vld_2xf64_backwards,
        neon_vst_2xf64,
        neon_vst_2xf64_backwards,
        neon_broadcast_2xf64,
        neon_broadcast_2xf64_scalar,
        neon_vfmadd_2xf64_2xf64,
        neon_vfmadd_reduce_2xf64_2xf64,
        neon_zero_2xf64,
        neon_assoc_reduce_add_instr_2xf64,
        neon_assoc_reduce_add_instr_2xf64_buffer,
        neon_vmul_2xf64,
        neon_vadd_2xf64,
        neon_reduce_vadd_2xf64,
        neon_reg_copy_2xf64,
        neon_vneg_2xf64,
        neon_convert_f32_lower_to_f64,
        neon_convert_f32_upper_to_f64,
    ],
    patterns=[fma_rule],
)
