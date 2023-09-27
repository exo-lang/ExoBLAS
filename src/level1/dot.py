from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import (
    apply_to_block,
    hoist_stmt,
    interleave_execution,
    vectorize_to_loops,
)
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params


### EXO_LOC ALGORITHM START ###
@proc
def dot_template(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]


### EXO_LOC ALGORITHM END ###


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


### EXO_LOC SCHEDULE START ###
def schedule_dot_stride_1(dot, params):
    dot = generate_stride_1_proc(dot, params.precision)
    main_loop = dot.find_loop("i")
    dot = stage_mem(dot, dot.body(), "result", "result_")
    dot = blas_vectorize(dot, main_loop, params)
    dot = unroll_loop(dot, dot.find_loop("ioi"))
    dot = unroll_loop(dot, dot.find_loop("ioi"))
    tail_loop = dot.find_loop("ii")
    dot = vectorize_to_loops(
        dot, tail_loop, params.vec_width, params.mem_type, params.precision
    )
    dot = replace_all(
        dot,
        [
            mm256_prefix_load_ps,
            mm256_prefix_load_pd,
            mm256_prefix_fmadd_ps,
            mm256_prefix_fmadd_pd,
        ],
    )
    return simplify(dot)


template_sched_list = [
    (dot_template, schedule_dot_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, Level_1_Params(precision=precision))
        export_exo_proc(globals(), proc_stride_1)
### EXO_LOC SCHEDULE END ###
