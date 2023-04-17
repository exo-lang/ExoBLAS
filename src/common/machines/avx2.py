from __future__ import annotations

from exo.platforms.x86 import *

from .machine import MachineParameters


@instr("{dst_data} = _mm256_maskz_loadu_ps(((1 << {N}) - 1), &{src_data});")
def mm256_maskz_loadu_ps(N: size, dst: [f32][8] @ AVX2, src: [f32][N] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 8

    for i in seq(0, N):
        dst[i] = src[i]


@instr(
    """
    {{
        __m256 tmp = _mm256_hadd_ps({x_data}, {x_data});
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        {result_data} += _mm256_cvtss_f32(tmp);
    }}
    """
)
def avx2_assoc_reduce_add_ps_buffer(x: [f32][8] @ AVX2, result: [f32][1]):
    # WARNING: This instruction assumes float addition associativity
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1

    for i in seq(0, 8):
        result[0] += x[i]


@instr(
    """
    {{
        __m256d tmp = _mm256_hadd_pd({x_data}, {x_data});
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        {result_data} += _mm256_cvtsd_f64(tmp);
    }}
    """
)
def avx2_assoc_reduce_add_pd_buffer(x: [f64][4] @ AVX2, result: [f64][1]):
    # WARNING: This instruction assumes float addition associativity
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1

    for i in seq(0, 4):
        result[0] += x[i]


@instr("{dst_data} = _mm256_loadu_ps(&{src_data});")
def avx2_loadu_ps_backwards(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[7 - i]


@instr("{dst_data} = _mm256_loadu_pd(&{src_data});")
def avx2_loadu_pd_backwards(dst: [f64][4] @ AVX2, src: [f64][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[3 - i]


@instr("_mm256_storeu_ps(&{dst_data}, {src_data});")
def avx2_storeu_ps_backwards(dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[7 - i] = src[i]


@instr("_mm256_storeu_pd(&{dst_data}, {src_data});")
def avx2_storeu_pd_backwards(dst: [f64][4] @ DRAM, src: [f64][4] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[3 - i] = src[i]


Machine = MachineParameters(
    name="avx2",
    mem_type=AVX2,
    n_vec_registers=16,
    vec_width=8,
    vec_units=2,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=mm256_loadu_ps,
    load_backwards_instr_f32=avx2_loadu_ps_backwards,
    store_instr_f32=mm256_storeu_ps,
    store_backwards_instr_f32=avx2_storeu_ps_backwards,
    broadcast_instr_f32=mm256_broadcast_ss,
    broadcast_scalar_instr_f32=mm256_broadcast_ss_scalar,
    fmadd_instr_f32=mm256_fmadd_ps,
    set_zero_instr_f32=mm256_setzero_ps,
    assoc_reduce_add_instr_f32=avx2_assoc_reduce_add_ps,
    assoc_reduce_add_f32_buffer=avx2_assoc_reduce_add_ps_buffer,
    mul_instr_f32=mm256_mul_ps,
    add_instr_f32=mm256_add_ps,
    reduce_add_wide_instr_f32=avx2_reduce_add_wide_ps,
    reg_copy_instr_f32=avx2_reg_copy_ps,
    sign_instr_f32=avx2_sign_ps,
    select_instr_f32=avx2_select_ps,
    load_instr_f64=mm256_loadu_pd,
    load_backwards_instr_f64=avx2_loadu_pd_backwards,
    store_instr_f64=mm256_storeu_pd,
    store_backwards_instr_f64=avx2_storeu_pd_backwards,
    broadcast_instr_f64=mm256_broadcast_sd,
    broadcast_scalar_instr_f64=mm256_broadcast_sd_scalar,
    fmadd_instr_f64=mm256_fmadd_pd,
    set_zero_instr_f64=mm256_setzero_pd,
    assoc_reduce_add_instr_f64=avx2_assoc_reduce_add_pd,
    mul_instr_f64=mm256_mul_pd,
    add_instr_f64=mm256_add_pd,
    reduce_add_wide_instr_f64=avx2_reduce_add_wide_pd,
    assoc_reduce_add_f64_buffer=avx2_assoc_reduce_add_pd_buffer,
    reg_copy_instr_f64=avx2_reg_copy_pd,
    sign_instr_f64=avx2_sign_pd,
    select_instr_f64=avx2_select_pd,
    convert_f32_lower_to_f64=avx2_convert_f32_lower_to_f64,
    convert_f32_upper_to_f64=avx2_convert_f32_upper_to_f64,
)
