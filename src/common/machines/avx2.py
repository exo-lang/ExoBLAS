from __future__ import annotations

from exo.platforms.x86 import *

from .machine import MachineParameters

from composed_schedules import *


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


@instr(
    """{dst_data} = _mm256_loadu_ps(&{src_data});
{dst_data} = _mm256_permutevar8x32_ps({dst_data}, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
"""
)
def avx2_loadu_ps_backwards(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[7 - i]


@instr(
    """
{dst_data} = _mm256_loadu_pd(&{src_data});
{dst_data} = _mm256_permute2f128_pd({dst_data}, {dst_data}, 1);
{dst_data} = _mm256_permute_pd({dst_data}, 1 + 4);
"""
)
def avx2_loadu_pd_backwards(dst: [f64][4] @ AVX2, src: [f64][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[3 - i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_maskload_ps(&{src_data}, cmp);
    {dst_data} = _mm256_permutevar8x32_ps({dst_data}, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}}
"""
)
def avx2_prefix_load_ps_backwards(
    dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM, bound: size
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 8
    for i in seq(0, 8):
        if i < bound:
            dst[i] = src[7 - i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi64x(0, 1, 2, 3);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    {dst_data} = _mm256_maskload_pd(&{src_data}, cmp);
    {dst_data} = _mm256_permute2f128_pd({dst_data}, {dst_data}, 1);
    {dst_data} = _mm256_permute_pd({dst_data}, 1 + 4);
}}
"""
)
def avx2_prefix_load_pd_backwards(
    dst: [f64][4] @ AVX2, src: [f64][4] @ DRAM, bound: size
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 4
    for i in seq(0, 4):
        if i < bound:
            dst[i] = src[3 - i]


@instr(
    """
{{
__m256 tmp = _mm256_permutevar8x32_ps({src_data}, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
_mm256_storeu_ps(&{dst_data}, tmp);
}}
"""
)
def avx2_storeu_ps_backwards(dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[7 - i] = src[i]


@instr(
    """
{{
__m256d tmp = _mm256_permute2f128_pd({src_data}, {src_data}, 1);
tmp = _mm256_permute_pd(tmp, 1 + 4);
_mm256_storeu_pd(&{dst_data}, tmp);
}}
"""
)
def avx2_storeu_pd_backwards(dst: [f64][4] @ DRAM, src: [f64][4] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[3 - i] = src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 tmp = _mm256_permutevar8x32_ps({src_data}, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    _mm256_maskstore_ps(&{dst_data}, cmp, tmp);
    }}
    """
)
def avx2_prefix_store_ps_backwards(
    dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2, bound: size
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 8
    for i in seq(0, 8):
        if i < bound:
            dst[7 - i] = src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi64x(0, 1, 2, 3);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d tmp = _mm256_permute2f128_pd({src_data}, {src_data}, 1);
    tmp = _mm256_permute_pd(tmp, 1 + 4);
    _mm256_maskstore_pd(&{dst_data}, cmp, tmp);
    }}
    """
)
def avx2_prefix_store_pd_backwards(
    dst: [f64][4] @ DRAM, src: [f64][4] @ AVX2, bound: size
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 4
    for i in seq(0, 4):
        if i < bound:
            dst[3 - i] = src[i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_maskload_ps(&{src_data}, cmp);
}}
"""
)
def mm256_prefix_load_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM, bound: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 8
    for i in seq(0, 8):
        if i < bound:
            dst[i] = src[i]


@instr(
    """
       {{
            __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
            __m256i prefix = _mm256_set1_epi64x({bound});
            __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
            {dst_data} = _mm256_maskload_pd(&{src_data}, cmp);
       }}
       """
)
def mm256_prefix_load_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ DRAM, bound: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 4
    for i in seq(0, 4):
        if i < bound:
            dst[i] = src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&{dst_data}, cmp, {src_data});
    }}
    """
)
def mm256_prefix_store_ps(dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 8
    for i in seq(0, 8):
        if i < bound:
            dst[i] = src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    _mm256_maskstore_pd(&{dst_data}, cmp, {src_data});
    }}
    """
)
def mm256_prefix_store_pd(dst: [f64][4] @ DRAM, src: [f64][4] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 4
    for i in seq(0, 4):
        if i < bound:
            dst[i] = src[i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), {src1_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_fmadd_ps(prefixed_src1, {src2_data}, {dst_data});
}}
"""
)
def mm256_prefix_fmadd_ps(
    dst: [f32][8] @ AVX2, src1: [f32][8] @ AVX2, src2: [f32][8] @ AVX2, bound: size
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 8

    for i in seq(0, 8):
        if i < bound:
            dst[i] += src1[i] * src2[i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {dst_data});
}}
"""
)
def mm256_prefix_fmadd_pd(
    dst: [f64][4] @ AVX2, src1: [f64][4] @ AVX2, src2: [f64][4] @ AVX2, bound: size
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1
    assert bound <= 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] += src1[i] * src2[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src = _mm256_blendv_ps (_mm256_setzero_ps(), {src_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_add_ps(prefixed_src, {dst_data});
    }}
    """
)
def avx2_prefix_reduce_add_wide_ps(
    dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2, bound: size
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 8):
        if i < bound:
            dst[i] += src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src = _mm256_blendv_pd (_mm256_setzero_pd(), {src_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_add_pd(prefixed_src, {dst_data});
    }}
    """
)
def avx2_prefix_reduce_add_wide_pd(
    dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2, bound: size
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        if i < bound:
            dst[i] += src[i]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {out_data} = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss(&{val_data}), _mm256_castsi256_ps(cmp));
    }}
    """
)
def mm256_prefix_broadcast_ss(out: [f32][8] @ AVX2, val: [f32][1], bound: size):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        if i < bound:
            out[i] = val[0]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    {out_data} = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd(&{val_data}), _mm256_castsi256_pd(cmp));
    }}
    """
)
def mm256_prefix_broadcast_sd(out: [f64][4] @ AVX2, val: [f64][1], bound: size):
    assert stride(out, 0) == 1

    for i in seq(0, 4):
        if i < bound:
            out[i] = val[0]


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {out_data} = _mm256_blendv_ps (_mm256_setzero_ps(), _mm256_broadcast_ss({val_data}), _mm256_castsi256_ps(cmp));
    }}
    """
)
def mm256_prefix_broadcast_ss_scalar(out: [f32][8] @ AVX2, val: f32, bound: size):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        if i < bound:
            out[i] = val


@instr(
    """
    {{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    {out_data} = _mm256_blendv_pd (_mm256_setzero_pd(), _mm256_broadcast_sd({val_data}), _mm256_castsi256_pd(cmp));
    }}
    """
)
def mm256_prefix_broadcast_sd_scalar(out: [f64][4] @ AVX2, val: f64, bound: size):
    assert stride(out, 0) == 1

    for i in seq(0, 4):
        if i < bound:
            out[i] = val


@instr(
    "{dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));"
)
def avx2_abs_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = select(0.0, src[i], src[i], -src[i])


@instr(
    """
{dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd (_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
"""
)
def avx2_abs_pd(dst: [f64][8] @ AVX2, src: [f64][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = select(0.0, src[i], src[i], -src[i])


@instr(
    """
{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_abs = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
{dst_data} = _mm256_blendv_ps ({dst_data}, src_abs, _mm256_castsi256_ps(cmp));
}}
"""
)
def avx2_prefix_abs_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 8
    for i in seq(0, 8):
        if i < bound:
            dst[i] = select(0.0, src[i], src[i], -src[i])


@instr(
    """
{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_abs = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
{dst_data} = _mm256_blendv_pd ({dst_data}, src_abs, _mm256_castsi256_pd(cmp));
}}
"""
)
def avx2_prefix_abs_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 4
    for i in seq(0, 4):
        if i < bound:
            dst[i] = select(0.0, src[i], src[i], -src[i])


@instr(
    """
{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 src_sign = _mm256_mul_ps({src_data}, _mm256_set1_ps(-1.0f));;
{dst_data} = _mm256_blendv_ps ({dst_data}, src_sign, _mm256_castsi256_ps(cmp));
}}
"""
)
def avx2_prefix_sign_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 8

    for i in seq(0, 8):
        if i < bound:
            dst[i] = -src[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d src_sign = _mm256_mul_pd({src_data}, _mm256_set1_pd(-1.0f));
{dst_data} = _mm256_blendv_pd ({dst_data}, src_sign, _mm256_castsi256_pd(cmp));
}}
"""
)
def avx2_prefix_sign_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound <= 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] = -src[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, mul, _mm256_castsi256_ps(cmp));
}}
"""
)
def mm256_prefix_mul_ps(
    out: [f32][8] @ AVX2, x: [f32][8] @ AVX2, y: [f32][8] @ AVX2, bound: size
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    assert bound <= 8

    for i in seq(0, 8):
        if i < bound:
            out[i] = x[i] * y[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d mul = _mm256_mul_pd({x_data}, {y_data});
{out_data} = _mm256_blendv_pd ({out_data}, mul, _mm256_castsi256_pd(cmp));
}}
"""
)
def mm256_prefix_mul_pd(
    out: [f64][4] @ AVX2, x: [f64][4] @ AVX2, y: [f64][4] @ AVX2, bound: size
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    assert bound <= 4

    for i in seq(0, 4):
        if i < bound:
            out[i] = x[i] * y[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, add, _mm256_castsi256_ps(cmp));
}}
"""
)
def mm256_prefix_add_ps(
    out: [f32][8] @ AVX2, x: [f32][8] @ AVX2, y: [f32][8] @ AVX2, bound: size
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    assert bound <= 8

    for i in seq(0, 8):
        if i < bound:
            out[i] = x[i] + y[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
__m256d add = _mm256_add_pd({x_data}, {y_data});
{out_data} = _mm256_blendv_pd ({out_data}, add, _mm256_castsi256_pd(cmp));
}}
"""
)
def mm256_prefix_add_pd(
    out: [f64][4] @ AVX2, x: [f64][4] @ AVX2, y: [f64][4] @ AVX2, bound: size
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1
    assert bound <= 4

    for i in seq(0, 4):
        if i < bound:
            out[i] = x[i] + y[i]


@instr(
    """
{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
{dst_data} = _mm256_blendv_ps ({dst_data}, _mm256_setzero_ps(), _mm256_castsi256_ps(cmp));
}}
"""
)
def mm256_prefix_setzero_ps(dst: [f32][8] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert bound <= 8

    for i in seq(0, 8):
        if i < bound:
            dst[i] = 0.0


@instr(
    """
{{
__m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi64x({bound});
__m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
{dst_data} = _mm256_blendv_pd ({dst_data}, _mm256_setzero_pd(), _mm256_castsi256_pd(cmp));
}}
"""
)
def mm256_prefix_setzero_pd(dst: [f64][4] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert bound <= 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] = 0.0


mm256_fmadd_reduce_ps = rename(mm256_fmadd_ps, "mm256_fmadd_reduce_ps")
mm256_fmadd_reduce_pd = rename(mm256_fmadd_pd, "mm256_fmadd_reduce_pd")
mm256_prefix_fmadd_reduce_ps = rename(
    mm256_prefix_fmadd_ps, "mm256_prefix_fmadd_reduce_ps"
)
mm256_prefix_fmadd_reduce_pd = rename(
    mm256_prefix_fmadd_pd, "mm256_prefix_fmadd_reduce_pd"
)


@instr("{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});")
def mm256_fmadd_ps(
    dst: [f32][8] @ AVX2,
    src1: [f32][8] @ AVX2,
    src2: [f32][8] @ AVX2,
    src3: [f32][8] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src1[i] * src2[i] + src3[i]


@instr("{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});")
def mm256_fmadd_pd(
    dst: [f64][4] @ AVX2,
    src1: [f64][4] @ AVX2,
    src2: [f64][4] @ AVX2,
    src3: [f64][4] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src1[i] * src2[i] + src3[i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    __m256 prefixed_src1 = _mm256_blendv_ps (_mm256_setzero_ps(), {src1_data}, _mm256_castsi256_ps(cmp));
    {dst_data} = _mm256_fmadd_ps(prefixed_src1, {src2_data}, {src3_data});
}}
"""
)
def mm256_prefix_fmadd_ps(
    dst: [f32][8] @ AVX2,
    src1: [f32][8] @ AVX2,
    src2: [f32][8] @ AVX2,
    src3: [f32][8] @ AVX2,
    bound: size,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1
    assert bound < 8

    for i in seq(0, 8):
        if i < bound:
            dst[i] = src1[i] * src2[i] + src3[i]


@instr(
    """
{{
    __m256i indices = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi64x({bound});
    __m256i cmp = _mm256_cmpgt_epi64(prefix, indices);
    __m256d prefixed_src1 = _mm256_blendv_pd(_mm256_setzero_pd(), {src1_data}, _mm256_castsi256_pd(cmp));
    {dst_data} = _mm256_fmadd_pd(prefixed_src1, {src2_data}, {src3_data});
}}
"""
)
def mm256_prefix_fmadd_pd(
    dst: [f64][4] @ AVX2,
    src1: [f64][4] @ AVX2,
    src2: [f64][4] @ AVX2,
    src3: [f64][4] @ AVX2,
    bound: size,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1
    assert bound < 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] = src1[i] * src2[i] + src3[i]


@instr("{dst_data} = _mm256_cvtps_pd(_mm_loadu_ps(&{src_data}));")
def avx2_fused_load_cvtps_pd(dst: [f64][4] @ AVX2, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr(
    """
{{
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    __m128i prefix = _mm_set1_epi32({bound});
    __m128i cmp = _mm_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_cvtps_pd(_mm_maskload_ps(&{src_data}, cmp));
}}
"""
)
def avx2_prefix_fused_load_cvtps_pd(
    dst: [f64][4] @ AVX2, src: [f32][4] @ DRAM, bound: size
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound < 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] = src[i]


@instr("{dst_data} = {src_data};")
def avx2_prefix_reg_copy_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound < 8

    for i in seq(0, 8):
        if i < bound:
            dst[i] = src[i]


@instr("{dst_data} = {src_data};")
def avx2_prefix_reg_copy_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2, bound: size):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    assert bound < 4

    for i in seq(0, 4):
        if i < bound:
            dst[i] = src[i]


Machine = MachineParameters(
    name="avx2",
    mem_type=AVX2,
    n_vec_registers=16,
    f32_vec_width=8,
    vec_units=2,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=mm256_loadu_ps,
    load_backwards_instr_f32=avx2_loadu_ps_backwards,
    prefix_load_instr_f32=mm256_prefix_load_ps,
    prefix_load_backwards_instr_f32=avx2_prefix_load_ps_backwards,
    store_instr_f32=mm256_storeu_ps,
    store_backwards_instr_f32=avx2_storeu_ps_backwards,
    prefix_store_instr_f32=mm256_prefix_store_ps,
    prefix_store_backwards_instr_f32=avx2_prefix_store_ps_backwards,
    broadcast_instr_f32=mm256_broadcast_ss,
    broadcast_scalar_instr_f32=mm256_broadcast_ss_scalar,
    prefix_broadcast_instr_f32=mm256_prefix_broadcast_ss,
    prefix_broadcast_scalar_instr_f32=mm256_prefix_broadcast_ss_scalar,
    fmadd_instr_f32=mm256_fmadd_ps,
    prefix_fmadd_instr_f32=mm256_prefix_fmadd_ps,
    fmadd_reduce_instr_f32=mm256_fmadd_reduce_ps,
    prefix_fmadd_reduce_instr_f32=mm256_prefix_fmadd_reduce_ps,
    set_zero_instr_f32=mm256_setzero_ps,
    prefix_set_zero_instr_f32=mm256_prefix_setzero_ps,
    assoc_reduce_add_instr_f32=avx2_assoc_reduce_add_ps,
    assoc_reduce_add_f32_buffer=avx2_assoc_reduce_add_ps_buffer,
    mul_instr_f32=mm256_mul_ps,
    prefix_mul_instr_f32=mm256_prefix_mul_ps,
    add_instr_f32=mm256_add_ps,
    prefix_add_instr_f32=mm256_prefix_add_ps,
    reduce_add_wide_instr_f32=avx2_reduce_add_wide_ps,
    prefix_reduce_add_wide_instr_f32=avx2_prefix_reduce_add_wide_ps,
    reg_copy_instr_f32=avx2_reg_copy_ps,
    prefix_reg_copy_instr_f32=avx2_prefix_reg_copy_ps,
    sign_instr_f32=avx2_sign_ps,
    prefix_sign_instr_f32=avx2_prefix_sign_ps,
    select_instr_f32=avx2_select_ps,
    abs_instr_f32=avx2_abs_ps,
    prefix_abs_instr_f32=avx2_prefix_abs_ps,
    load_instr_f64=mm256_loadu_pd,
    load_backwards_instr_f64=avx2_loadu_pd_backwards,
    prefix_load_instr_f64=mm256_prefix_load_pd,
    prefix_load_backwards_instr_f64=avx2_prefix_load_pd_backwards,
    store_instr_f64=mm256_storeu_pd,
    store_backwards_instr_f64=avx2_storeu_pd_backwards,
    prefix_store_instr_f64=mm256_prefix_store_pd,
    prefix_store_backwards_instr_f64=avx2_prefix_store_pd_backwards,
    broadcast_instr_f64=mm256_broadcast_sd,
    broadcast_scalar_instr_f64=mm256_broadcast_sd_scalar,
    prefix_broadcast_instr_f64=mm256_prefix_broadcast_sd,
    prefix_broadcast_scalar_instr_f64=mm256_prefix_broadcast_sd_scalar,
    fmadd_instr_f64=mm256_fmadd_pd,
    prefix_fmadd_instr_f64=mm256_prefix_fmadd_pd,
    fmadd_reduce_instr_f64=mm256_fmadd_reduce_pd,
    prefix_fmadd_reduce_instr_f64=mm256_prefix_fmadd_reduce_pd,
    set_zero_instr_f64=mm256_setzero_pd,
    prefix_set_zero_instr_f64=mm256_prefix_setzero_pd,
    assoc_reduce_add_instr_f64=avx2_assoc_reduce_add_pd,
    mul_instr_f64=mm256_mul_pd,
    prefix_mul_instr_f64=mm256_prefix_mul_pd,
    add_instr_f64=mm256_add_pd,
    prefix_add_instr_f64=mm256_prefix_add_pd,
    reduce_add_wide_instr_f64=avx2_reduce_add_wide_pd,
    prefix_reduce_add_wide_instr_f64=avx2_prefix_reduce_add_wide_pd,
    assoc_reduce_add_f64_buffer=avx2_assoc_reduce_add_pd_buffer,
    reg_copy_instr_f64=avx2_reg_copy_pd,
    prefix_reg_copy_instr_f64=avx2_prefix_reg_copy_pd,
    sign_instr_f64=avx2_sign_pd,
    prefix_sign_instr_f64=avx2_prefix_sign_pd,
    select_instr_f64=avx2_select_pd,
    abs_instr_f64=avx2_abs_pd,
    prefix_abs_instr_f64=avx2_prefix_abs_pd,
    convert_f32_lower_to_f64=avx2_convert_f32_lower_to_f64,
    convert_f32_upper_to_f64=avx2_convert_f32_upper_to_f64,
    fused_load_cvt_f32_f64=avx2_fused_load_cvtps_pd,
    prefix_fused_load_cvt_f32_f64=avx2_prefix_fused_load_cvtps_pd,
)
