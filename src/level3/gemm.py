from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from format_options import *

import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)


def gemm_test():
    @proc
    def exo_sgemm_test(
        M: size,
        N: size,
        K: size,
        C: f32[M, N] @ DRAM,
        A: f32[M, K] @ DRAM,
        B: f32[K, N] @ DRAM,
    ):
        for i in seq(0, M):
            for k in seq(0, K):
                for j in seq(0, N):
                    C[i, j] += A[i, k] * B[k, j]

    return exo_sgemm_test

VECTORIZATION_INTERLEAVE_FACTOR = 2
ROWS_INTERLEAVE_FACTOR = 4
VEC_W = C.Machine.vec_width

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.set_zero_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.assoc_reduce_add_instr_f32,
    C.Machine.assoc_reduce_add_f32_buffer,
    C.Machine.reg_copy_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

#TODO: Do more compute in the innermost loop
#TODO: Check whether there is a reuse potential if we interleave 
#      the outer loop all the way in after casting the computation as gemv
#      and before blocking

exo_sgemm_test = gemm_test()
loop_cursor = exo_sgemm_test.find_loop("j")
# Inner loop vectorization
exo_sgemm_test = vectorize(exo_sgemm_test, loop_cursor, C.Machine.vec_width, AVX2, "f32")
# TODO: Interleave execution here

# Middle loop interleaving to turn inner two loops into gemv
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("k"),
    exo_sgemm_test.find_loop("jo"),
    ROWS_INTERLEAVE_FACTOR,
)
exo_sgemm_test = stage_mem(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("ki"),
    f"C[i, 8 * jo:8 + 8 * jo]",
    f"CReg",
)
exo_sgemm_test = set_memory(exo_sgemm_test, "CReg:_", AVX2)

exo_sgemm_test = unroll_loop(exo_sgemm_test, "ki")  
exo_sgemm_test = simplify(exo_sgemm_test)
exo_sgemm_test = reorder_loops(exo_sgemm_test, exo_sgemm_test.find_loop("ki"))
exo_sgemm_test = replace_all(exo_sgemm_test, f32_instructions)

# Three-level blcoking
B1 = 2048
B2 = 256
B3 = 64
B4 = B3 // 2

exo_sgemm_test = divide_loop(exo_sgemm_test, "jo", B1, ["joo", "joi"], tail="cut")
exo_sgemm_test = simplify(exo_sgemm_test)



exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("ko"),
    exo_sgemm_test.find_loop("joo"),
    B1,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("i"),
    exo_sgemm_test.find_loop("koo"),
    B1,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("ii"),
    exo_sgemm_test.find_loop("joo"),
    B1,
)

exo_sgemm_test = simplify(exo_sgemm_test)

exo_sgemm_test = divide_loop(exo_sgemm_test, "joi", B2, ["joio", "joii"], tail="cut")
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("koi"),
    exo_sgemm_test.find_loop("joio"),
    B2,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("ii"),
    exo_sgemm_test.find_loop("koio"),
    B2,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("iii"),
    exo_sgemm_test.find_loop("joio"),
    B2,
)

exo_sgemm_test = divide_loop(exo_sgemm_test, "joii", B3, ["joiio", "joiii"], tail="cut")
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("koii"),
    exo_sgemm_test.find_loop("joiio"),
    B3,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("iii"),
    exo_sgemm_test.find_loop("koiio"),
    B3,
)
exo_sgemm_test = interleave_outer_loop_with_inner_loop(
    exo_sgemm_test,
    exo_sgemm_test.find_loop("iiii"),
    exo_sgemm_test.find_loop("joiio"),
    B3,
)
exo_sgemm_test = simplify(exo_sgemm_test)


__all__ = [exo_sgemm_test.name()]