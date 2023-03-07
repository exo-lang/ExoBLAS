from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def sdsdot_template(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    d_result: f64
    d_result = sb
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_result += d_x * d_y
    result = d_result
    
def schedule_sdsdot_stride_1(VEC_W, memory, instructions):
    simple_stride_1 = rename(sdsdot_template, "exo_sdsdot_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")
    
    simple_stride_1 = divide_loop(simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail = "cut")
    
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(stage_mem(simple_stride_1, "for io in _:_", "d_result", "resultReg", accum=True))
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _")
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").before())
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("for io in _:_").after())
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")

    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"x[{VEC_W} * io: {VEC_W} * io + {VEC_W}]", "xReg")
    simple_stride_1 = stage_mem(simple_stride_1, "for ii in _:_ #1", f"y[{VEC_W} * io: {VEC_W} * io + {VEC_W}]", "yReg")
    
    simple_stride_1 = expand_dim(simple_stride_1, "d_x", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "d_x")
    simple_stride_1 = expand_dim(simple_stride_1, "d_y", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "d_y")
    
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("d_y[_] = _").after())
    simple_stride_1 = fission(simple_stride_1, simple_stride_1.find("d_y[_] = _").before())
    
    simple_stride_1 = cut_loop(simple_stride_1, simple_stride_1.find("resultReg[_] = _").parent(), VEC_W // 2)
    simple_stride_1 = cut_loop(simple_stride_1, simple_stride_1.find("d_x[_] = _").parent(), VEC_W // 2)
    simple_stride_1 = cut_loop(simple_stride_1, simple_stride_1.find("d_y[_] = _").parent(), VEC_W // 2)
    simple_stride_1 = cut_loop(simple_stride_1, simple_stride_1.find("resultReg += _").parent(), VEC_W // 2)
    simple_stride_1 = cut_loop(simple_stride_1, simple_stride_1.find("d_result += _").parent(), VEC_W // 2)
    
    simple_stride_1 = divide_dim(simple_stride_1, "d_x", 0, VEC_W // 2)
    simple_stride_1 = divide_dim(simple_stride_1, "d_y", 0, VEC_W // 2)
    simple_stride_1 = divide_dim(simple_stride_1, "resultReg", 0, VEC_W // 2)
    
    for buffer in ["xReg", "yReg", "resultReg", "d_x", "d_y"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
    
    simple_stride_1 = simplify(simple_stride_1)
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    
    # TODO: remove once set_memory takes allocation cursor
    for p in [("d_x", "x"), ("d_y", "y")]:
        buffer = p[0]
        rhs = p[1]
        tail_loop_block = simple_stride_1.find(f"{buffer} = {rhs}[_]").expand(2)
        simple_stride_1 = stage_mem(simple_stride_1, tail_loop_block, buffer, f"{buffer}_tmp")
        tmp_buffer = simple_stride_1.find(f"{buffer}_tmp : _")
        xReg_buffer = tmp_buffer.prev()
        simple_stride_1 = reuse_buffer(simple_stride_1, tmp_buffer, xReg_buffer)
        simple_stride_1 = set_memory(simple_stride_1, f"{buffer}_tmp", DRAM)
    
    return simple_stride_1

@instr("{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 0));")
def avx2_convert_f32_lower_to_f64(dst: [f64][4] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    
    for i in seq(0, 4):
        dst[i] = src[i]
        
@instr("{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 1));")
def avx2_convert_f32_upper_to_f64(dst: [f64][4] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    
    for i in seq(0, 4):
        dst[i] = src[4 + i]

exo_sdsdot_stride_any = sdsdot_template
exo_sdsdot_stride_any = rename(exo_sdsdot_stride_any, "exo_sdsdot_stride_any")

instructions = [
                C.Machine.load_instr_f32,
                C.Machine.store_instr_f32,
                C.Machine.set_zero_instr_f64,
                C.Machine.fmadd_instr_f64,
                C.Machine.assoc_reduce_add_instr_f64,
                ]

if None not in instructions:
    exo_sdsdot_stride_1 = schedule_sdsdot_stride_1(C.Machine.vec_width, C.Machine.mem_type, instructions)
    if C.Machine.mem_type is AVX2:
        for i in range(2):
            exo_sdsdot_stride_1 = replace(exo_sdsdot_stride_1, "for ii in _:_", avx2_convert_f32_lower_to_f64)
            exo_sdsdot_stride_1 = replace(exo_sdsdot_stride_1, "for ii in _:_", avx2_convert_f32_upper_to_f64)
else:
    exo_sdsdot_stride_1 = sdsdot_template
    exo_sdsdot_stride_1 = rename(exo_sdsdot_stride_1, "exo_sdsdot_stride_1")

entry_points = [exo_sdsdot_stride_any, exo_sdsdot_stride_1]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]