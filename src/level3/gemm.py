from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

from composed_schedules import *
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    generate_stride_any_proc,
    export_exo_proc,
)
from parameters import Level_3_Params


@proc
def gemm_matmul_template(M: size, N: size, K: size, A: R[M, K], B: R[K, N], C: R[M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_gemm_matmul(gemm, params):
    gemm = generate_stride_any_proc(gemm, params.precision)
    return simplify(gemm)


template_sched_list = [
    (gemm_matmul_template, schedule_gemm_matmul),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_1 = sched(
            template,
            Level_3_Params(precision=precision),
        )
        export_exo_proc(globals(), proc_stride_1)
