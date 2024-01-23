from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    auto_divide_loop,
    scalar_to_simd,
    vectorize,
    parallelize_reduction,
    stage_expr,
    auto_stage_mem,
)
from codegen_helpers import (
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
    bind_builtins_args,
)
from parameters import Level_1_Params


@instr("{dst_data} = _mm256_sad_epu8(&{src1_data}, &{src2_data});")
def mm256_sad_epu8(dst: [R][4] @ AVX2, src1: [i8][32] @ AVX2, src2: [i8][32] @ AVX2):
    for i in seq(0, 4):
        tmp: R @ AVX2
        tmp = 0.0
        for j in seq(0, 8):
            tmp += select(
                src1[i * 8 + j],
                src2[i * 8 + j],
                src1[i * 8 + j] - src2[i * 8 + j],
                src2[i * 8 + j] - src1[i * 8 + j],
            )
        dst[i] = tmp


@proc
def sad(n: size, x: i8[n] @ DRAM, y: i8[n] @ DRAM, result: R @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(y[i], x[i], x[i] - y[i], y[i] - x[i])


sad = sad.partial_eval(1024)
i_loop = sad.find_loop("i")

# blocking, binding reg
sad, loop_c = auto_divide_loop(sad, i_loop, 8, perfect=True)
sad, _ = auto_divide_loop(sad, i_loop, 4, perfect=True)
sad = parallelize_reduction(sad, sad.find("result += _"), AVX2, 3)
sad = stage_mem(sad, loop_c.inner_loop, "var0[ioi]", "tmp_reg", accum=True)
# sad = auto_stage_mem(sad, 'reg[ioi]', 'tmp_reg', accum=True)
sad = simplify(sad)
sad = stage_expr(sad, sad.find("tmp_reg"), "tmp_reg1", memory=AVX2)

# binding x and y
sad = auto_stage_mem(sad, sad.find("x[_]"), "xReg", n_lifts=2)
sad = auto_stage_mem(sad, sad.find("y[_]"), "yReg", n_lifts=2)
sad = simplify(sad)
sad = set_memory(sad, "xReg:_", AVX2)
sad = set_memory(sad, "yReg:_", AVX2)

# FIXME: for now set all memory to DRAM so that the codegen doesn't complain
sad = set_memory(sad, "xReg:_", DRAM)
sad = set_memory(sad, "yReg:_", DRAM)
sad = set_memory(sad, "var0:_", DRAM)
sad = set_memory(sad, "tmp_reg1:_", DRAM)
sad = set_memory(sad, "tmp_reg:_", DRAM)

# replace
# sad = replace_all_stmts(sad, [mm256_sad_epu8])
sad = bind_builtins_args(sad, sad.body(), "i8")

sad = simplify(sad)
