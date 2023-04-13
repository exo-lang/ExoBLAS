from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import vectorize, interleave_execution, parallelize_reduction, stage_expr

@proc
def exo_sasum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])
        
@proc
def exo_dasum(n: size, x: [f64][n] @ DRAM, result: f64 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])

def schedule_asum_stride_1_interleaved(asum, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    simple_stride_1 = rename(asum, asum.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
            
    simple_stride_1 = parallelize_reduction(simple_stride_1, simple_stride_1.find_loop("i"), "result",\
        VEC_W, INTERLEAVE_FACTOR // 2, memory, precision)
    
    lower_bound = f"{VEC_W} * im + {VEC_W * INTERLEAVE_FACTOR // 2} * io"
    simple_stride_1 = simplify(stage_mem(simple_stride_1, simple_stride_1.find_loop("im").body()[0], f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg"))
    
    simple_stride_1 = stage_expr(simple_stride_1, simple_stride_1.find("select(_)"), "selectReg")
    
    for buffer in ["xReg", "selectReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)
    
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    for i in range(4):
        select_cursor = simple_stride_1.find("select(_, _, _, _)")
        arg_cursor = select_cursor.args()[i]
        simple_stride_1 = bind_expr(simple_stride_1, [arg_cursor], f"tmp_{i}")

    simple_stride_1 = interleave_execution(simple_stride_1, simple_stride_1.find_loop("im"), INTERLEAVE_FACTOR // 2)
    simple_stride_1 = interleave_execution(simple_stride_1, simple_stride_1.find_loop("io"), 2)
    simple_stride_1 = simplify(simple_stride_1)
    
    return simple_stride_1

@instr("{dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));")
def avx2_abs_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = select(0.0, src[i], src[i], -src[i])
        
@instr(
"""
{dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd (_mm256_blend_epi32( _mm256_set1_epi32(0x7FFFFFFF),
                                                                                _mm256_set1_epi32(0xFFFFFFFF),
                                                                                1 + 4 + 16 + 64)));
""")
def avx2_abs_pd(dst: [f64][8] @ AVX2, src: [f64][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = select(0.0, src[i], src[i], -src[i])

INTERLEAVE_FACTOR = C.Machine.vec_units * 2 * 2

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sasum_stride_any = exo_sasum
for i in range(4):
    select_cursor = exo_sasum_stride_any.find("select(_, _, _, _)")
    arg_cursor = select_cursor.args()[i]
    exo_sasum_stride_any = bind_expr(exo_sasum_stride_any, [arg_cursor], f"tmp_{i}")
exo_sasum_stride_any = rename(exo_sasum_stride_any, exo_sasum_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                C.Machine.store_instr_f32, 
                C.Machine.assoc_reduce_add_instr_f32,
                C.Machine.set_zero_instr_f32,
                C.Machine.reduce_add_wide_instr_f32,
                avx2_abs_ps if C.Machine.mem_type is AVX2 else None]

if None not in f32_instructions:
    exo_sasum_stride_1 = schedule_asum_stride_1_interleaved(exo_sasum, C.Machine.vec_width, INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32")
else:
    exo_sasum_stride_1 = exo_sasum_stride_any
    exo_sasum_stride_1 = rename(exo_sasum_stride_1, "exo_sasum_stride_1")

#################################################
# Generate specialized kernels for f64s precision
#################################################

exo_dasum_stride_any = exo_dasum
for i in range(4):
    select_cursor = exo_dasum_stride_any.find("select(_, _, _, _)")
    arg_cursor = select_cursor.args()[i]
    exo_dasum_stride_any = bind_expr(exo_dasum_stride_any, [arg_cursor], f"tmp_{i}")
exo_dasum_stride_any = rename(exo_dasum_stride_any, exo_dasum_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                C.Machine.store_instr_f64, 
                C.Machine.assoc_reduce_add_instr_f64,
                C.Machine.set_zero_instr_f64,
                C.Machine.reduce_add_wide_instr_f64,
                avx2_abs_pd if C.Machine.mem_type is AVX2 else None]

if None not in f64_instructions:
    exo_dasum_stride_1 = schedule_asum_stride_1_interleaved(exo_dasum, C.Machine.vec_width // 2, INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64")
else:
    exo_dasum_stride_1 = exo_dasum_stride_any
    exo_dasum_stride_1 = rename(exo_dasum_stride_1, "exo_dasum_stride_1")

entry_points = [exo_sasum_stride_any, exo_sasum_stride_1, exo_dasum_stride_any, exo_dasum_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
