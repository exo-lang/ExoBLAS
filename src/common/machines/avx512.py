from __future__ import annotations

from exo.platforms.x86 import *
from exo.stdlib.scheduling import *

from .machine import MachineParameters


@instr("{C_data} = _mm512_mask_fmadd_ps({A_data}, ((1 << {N}) - 1), {B_data}, {C_data});")
def mm512_mask_fmadd_ps(
    N: size,
    A: [f32][16] @ AVX512,
    B: [f32][16] @ AVX512,
    C: [f32][16] @ AVX512,
):
    assert N >= 1
    assert N <= 16
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in seq(0, 16):
        if i < N:
            C[i] += A[i] * B[i]


@instr("{dst_data} = _mm512_set1_ps({src_data});")
def mm512_mask_set1_ps(
    N: size,
    dst: [f32][16] @ AVX512,
    src: [f32][1],
):
    assert N >= 1
    assert N <= 16
    assert stride(dst, 0) == 1

    for i in seq(0, 16):
        if i < N:
            dst[i] = src[0]


mm512_fmadd_reduce_ps = rename(mm512_fmadd_ps, "mm512_fmadd_reduce_ps")
mm512_prefix_fmadd_reduce_ps = rename(mm512_mask_fmadd_ps, "mm512_prefix_fmadd_reduce_ps")


@instr("{dst_data} = _mm512_fmadd_ps({A_data}, {B_data}, {C_data});")
def mm512_fmadd_ps(
    dst: [f32][16] @ AVX512,
    A: [f32][16] @ AVX512,
    B: [f32][16] @ AVX512,
    C: [f32][16] @ AVX512,
):
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in seq(0, 16):
        dst[i] = C[i] + A[i] * B[i]


@instr("{dst_data} = _mm512_mask_fmadd_ps({A_data}, ((1 << {N}) - 1), {B_data}, {C_data});")
def mm512_mask_fmadd_ps(
    N: size,
    dst: [f32][16] @ AVX512,
    A: [f32][16] @ AVX512,
    B: [f32][16] @ AVX512,
    C: [f32][16] @ AVX512,
):
    assert N >= 1
    assert N <= 16
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in seq(0, 16):
        if i < N:
            dst[i] = C[i] + A[i] * B[i]


@instr("{dst_data} = _mm512_setzero_ps();")
def mm512_setzero_ps(dst: [f32][16] @ AVX512):
    assert stride(dst, 0) == 1

    for i in seq(0, 16):
        dst[i] = 0.0


@instr("{dst_data} = _mm512_setzero_ps();")
def mm512_prefix_setzero_ps(dst: [f32][16] @ AVX512, bound: size):
    assert stride(dst, 0) == 1
    assert bound <= 16

    for i in seq(0, 16):
        if i < bound:
            dst[i] = 0.0


@instr("{out_data} = _mm512_set1_ps (*{val_data});")
def mm512_set1_ps_scalar(out: [f32][16] @ AVX512, val: f32):
    assert stride(out, 0) == 1

    for i in seq(0, 16):
        out[i] = val


Machine = MachineParameters(
    name="avx512",
    mem_type=AVX512,
    n_vec_registers=32,
    f32_vec_width=16,
    vec_units=2,
    supports_predication=True,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=mm512_loadu_ps,
    load_backwards_instr_f32=None,
    prefix_load_instr_f32=mm512_maskz_loadu_ps,
    prefix_load_backwards_instr_f32=None,
    store_instr_f32=mm512_storeu_ps,
    store_backwards_instr_f32=None,
    prefix_store_backwards_instr_f32=None,
    prefix_store_instr_f32=mm512_mask_storeu_ps,
    broadcast_instr_f32=mm512_set1_ps,
    broadcast_scalar_instr_f32=mm512_set1_ps_scalar,
    prefix_broadcast_instr_f32=mm512_mask_set1_ps,
    prefix_broadcast_scalar_instr_f32=None,
    fmadd_instr_f32=mm512_fmadd_ps,
    prefix_fmadd_instr_f32=mm512_mask_fmadd_ps,
    fmadd_reduce_instr_f32=mm512_fmadd_reduce_ps,
    prefix_fmadd_reduce_instr_f32=mm512_prefix_fmadd_reduce_ps,
    set_zero_instr_f32=mm512_setzero_ps,
    prefix_set_zero_instr_f32=mm512_prefix_setzero_ps,
    assoc_reduce_add_instr_f32=None,
    mul_instr_f32=None,
    prefix_mul_instr_f32=None,
    add_instr_f32=None,
    prefix_add_instr_f32=None,
    reduce_add_wide_instr_f32=None,
    prefix_reduce_add_wide_instr_f32=None,
    reg_copy_instr_f32=None,
    prefix_reg_copy_instr_f32=None,
    sign_instr_f32=None,
    prefix_sign_instr_f32=None,
    select_instr_f32=None,
    assoc_reduce_add_f32_buffer=None,
    abs_instr_f32=None,
    prefix_abs_instr_f32=None,
    load_instr_f64=None,
    load_backwards_instr_f64=None,
    prefix_load_instr_f64=None,
    prefix_load_backwards_instr_f64=None,
    prefix_store_instr_f64=None,
    prefix_store_backwards_instr_f64=None,
    store_instr_f64=None,
    store_backwards_instr_f64=None,
    broadcast_instr_f64=None,
    broadcast_scalar_instr_f64=None,
    prefix_broadcast_instr_f64=None,
    prefix_broadcast_scalar_instr_f64=None,
    fmadd_instr_f64=None,
    prefix_fmadd_instr_f64=None,
    fmadd_reduce_instr_f64=None,
    prefix_fmadd_reduce_instr_f64=None,
    set_zero_instr_f64=None,
    prefix_set_zero_instr_f64=None,
    assoc_reduce_add_instr_f64=None,
    mul_instr_f64=None,
    prefix_mul_instr_f64=None,
    add_instr_f64=None,
    prefix_add_instr_f64=None,
    reduce_add_wide_instr_f64=None,
    prefix_reduce_add_wide_instr_f64=None,
    assoc_reduce_add_f64_buffer=None,
    reg_copy_instr_f64=None,
    prefix_reg_copy_instr_f64=None,
    sign_instr_f64=None,
    prefix_sign_instr_f64=None,
    select_instr_f64=None,
    abs_instr_f64=None,
    prefix_abs_instr_f64=None,
    convert_f32_lower_to_f64=None,
    convert_f32_upper_to_f64=None,
    fused_load_cvt_f32_f64=None,
    prefix_fused_load_cvt_f32_f64=None,
)
