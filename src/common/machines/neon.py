from __future__ import annotations

from exo.platforms.neon import *
from exo import instr

from .machine import MachineParameters


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
def neon_assoc_reduce_add_instr_4xf32_buffer(
    result: [f32][1] @ DRAM, x: [f32][4] @ Neon
):
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1
    for i in seq(0, 4):
        result[0] += x[i]


@instr("{result_data} += vaddvq_f64({x_data});")
def neon_assoc_reduce_add_instr_2xf64_buffer(
    result: [f64][1] @ DRAM, x: [f64][2] @ Neon
):
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


Machine = MachineParameters(
    name="neon",
    mem_type=Neon,
    n_vec_registers=32,
    vec_width=4,
    vec_units=4,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=neon_vld_4xf32,
    load_backwards_instr_f32=neon_vld_4xf32_backawrds,
    prefix_load_instr_f32=None,
    store_instr_f32=neon_vst_4xf32,
    store_backwards_instr_f32=neon_vst_4xf32_backawrds,
    prefix_store_instr_f32=None,
    broadcast_instr_f32=neon_broadcast_4xf32,
    broadcast_scalar_instr_f32=neon_broadcast_4xf32_scalar,
    prefix_broadcast_instr_f32=None,
    prefix_broadcast_scalar_instr_f32=None,
    fmadd_instr_f32=neon_vfmadd_4xf32_4xf32,
    prefix_fmadd_instr_f32=None,
    set_zero_instr_f32=neon_zero_4xf32,
    assoc_reduce_add_instr_f32=neon_assoc_reduce_add_instr_4xf32,
    mul_instr_f32=neon_vmul_4xf32,
    add_instr_f32=neon_vadd_4xf32,
    reduce_add_wide_instr_f32=neon_reduce_vadd_4xf32,
    prefix_reduce_add_wide_instr_f32=None,
    reg_copy_instr_f32=neon_reg_copy_4xf32,
    sign_instr_f32=neon_vneg_4xf32,
    select_instr_f32=None,
    assoc_reduce_add_f32_buffer=neon_assoc_reduce_add_instr_4xf32_buffer,
    abs_instr_f32=None,
    prefix_abs_instr_f32=None,
    load_instr_f64=neon_vld_2xf64,
    load_backwards_instr_f64=neon_vld_2xf64_backwards,
    prefix_load_instr_f64=None,
    store_instr_f64=neon_vst_2xf64,
    store_backwards_instr_f64=neon_vst_2xf64_backwards,
    prefix_store_instr_f64=None,
    broadcast_instr_f64=neon_broadcast_2xf64,
    broadcast_scalar_instr_f64=neon_broadcast_2xf64_scalar,
    prefix_broadcast_instr_f64=None,
    prefix_broadcast_scalar_instr_f64=None,
    fmadd_instr_f64=neon_vfmadd_2xf64_2xf64,
    prefix_fmadd_instr_f64=None,
    set_zero_instr_f64=neon_zero_2xf64,
    assoc_reduce_add_instr_f64=neon_assoc_reduce_add_instr_2xf64,
    assoc_reduce_add_f64_buffer=neon_assoc_reduce_add_instr_2xf64_buffer,
    mul_instr_f64=neon_vmul_2xf64,
    add_instr_f64=neon_vadd_2xf64,
    reduce_add_wide_instr_f64=neon_reduce_vadd_2xf64,
    prefix_reduce_add_wide_instr_f64=None,
    reg_copy_instr_f64=neon_reg_copy_2xf64,
    sign_instr_f64=neon_vneg_2xf64,
    select_instr_f64=None,
    abs_instr_f64=None,
    prefix_abs_instr_f64=None,
    convert_f32_lower_to_f64=neon_convert_f32_lower_to_f64,
    convert_f32_upper_to_f64=neon_convert_f32_upper_to_f64,
)
