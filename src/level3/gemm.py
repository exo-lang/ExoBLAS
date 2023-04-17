from __future__ import annotations

from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from format_options import *

import exo_blas_config as C


class GEMM:
    def __init__(
        self,
        machine: "MachineParameters",
        precision: str,
        K_blk: int,
        M_blk: int,
        N_blk: int,
        M_reg: int,
        N_reg: int,
        do_rename: bool = False,
        main: bool = True,
    ):

        ### Specialize for different precisions
        self.precision = precision
        self.prefix = "s" if precision == "f32" else "d"
        self.main = main

        ### GEMM PROCEDURES
        @proc
        def gemm_notranspose_noalpha(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * B[k, j]

        gemm_notranspose_noalpha = rename(
            gemm_notranspose_noalpha,
            f"{self.prefix}{gemm_notranspose_noalpha.name()}_{N_blk*2}_{M_blk}_{K_blk}",
        )
        gemm_notranspose_noalpha = self.specialize_gemm(
            gemm_notranspose_noalpha, self.precision, ["A", "B", "C"]
        )

        ### GEMM kernels
        self.microkernel = Microkernel(
            machine, M_reg, N_reg, K_blk, self.precision
        )  # TODO: Add precision args to microkernel, gebp, gepp
        self.gebp = GEBP_kernel(self.microkernel, M_blk, N_blk, self.precision)
        self.gepp = GEPP_kernel(self.gebp, self.precision)

        ### GEMM variants
        gemm_scheduled_notranspose_noalpha = self.schedule_gemm_notranspose_noalpha(
            gemm_notranspose_noalpha
        )

        ### Create entry points

        # Alpha = Beta = 1
        @proc
        def exo_gemm_notranspose_noalpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            gemm_scheduled_notranspose_noalpha(M, N, K, C, A, B)

        exo_gemm_notranspose_noalpha_nobeta = self.specialize_gemm(
            exo_gemm_notranspose_noalpha_nobeta, self.precision
        )

        self.entry_points = [
            exo_gemm_notranspose_noalpha_nobeta,
        ]

    def schedule_gemm_notranspose_noalpha(self, gemm_procedure: Procedure):
        gemm_scheduled = divide_loop(
            gemm_procedure,
            "k",
            self.microkernel.K_blk,
            ["ko", "ki"],
            tail="cut_and_guard",
        )
        gemm_scheduled = autofission(
            gemm_scheduled, gemm_scheduled.find("for ko in _:_ #0").after(), n_lifts=2
        )
        gemm_scheduled = reorder_loops(gemm_scheduled, "j ko")
        gemm_scheduled = reorder_loops(gemm_scheduled, "i ko")

        gemm_scheduled = divide_loop(
            gemm_scheduled, "j", self.gebp.N_blk, ["jo", "ji"], tail="cut"
        )
        gemm_scheduled = autofission(
            gemm_scheduled, gemm_scheduled.find("for jo in _:_ #0").after(), n_lifts=2
        )
        gemm_scheduled = reorder_loops(gemm_scheduled, "i jo")
        gemm_scheduled = reorder_loops(gemm_scheduled, "ko jo")
        gemm_scheduled = stage_mem(
            gemm_scheduled,
            "for i in _:_ #0",
            f"B[{self.gepp.K_blk} * ko:{self.gepp.K_blk} + {self.gepp.K_blk} * ko, {self.gepp.N_blk}*jo:{self.gepp.N_blk}*jo+{self.gepp.N_blk}]",
            "B_packed",
        )

        gemm_scheduled = replace(gemm_scheduled, "for i in _:_ #0", self.gepp.gepp_base)
        gemm_scheduled = call_eqv(
            gemm_scheduled,
            f"gepp_base_{self.gepp.this_id}(_)",
            self.gepp.gepp_scheduled,
        )
        return simplify(gemm_scheduled)

    def specialize_gemm(
        self,
        gemm: Procedure,
        precision: str,
        args: list[str] = ["A", "B", "C", "alpha", "beta"],
    ):

        prefix = "s" if precision == "f32" else "d"
        name = gemm.name().replace("exo_", "")
        specialized = rename(gemm, "exo_" + prefix + name)

        for arg in args:
            specialized = set_precision(specialized, arg, precision)

        if self.main:
            specialized = rename(specialized, specialized.name() + "_main")

        return specialized


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
n_blk = C.gemm.n_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg

#################################################
# Generate f32 kernels
#################################################

# sgemm_main = GEMM(C.Machine, "f32", k_blk, m_blk, n_blk, m_reg, n_reg)


square_blk_sizes = [2**i for i in range(5, 9)]
sgemm_square_kernels = [
    GEMM(C.Machine, "f32", blk, blk, blk, m_reg, n_reg, True, False)
    for blk in square_blk_sizes
]

n_blk = 8192
m_blk = 256
k_blk = 512
sgemm_main = GEMM(C.Machine, "f32", k_blk, m_blk, n_blk, m_reg, n_reg, False, True)

sgemm_entry_points = [p.name() for p in sgemm_main.entry_points]
exo_sgemm_notranspose_noalpha_nobeta_main = sgemm_main.entry_points[0]

__all__ = sgemm_entry_points
print(__all__)
