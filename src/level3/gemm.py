from __future__ import annotations
from os import abort

import sys
import getopt
import math
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *
from exo.libs.memories import DRAM_STATIC

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from format_options import *

from composed_schedules import tile_loops, auto_stage_mem

import exo_blas_config as C

"""
TODO:
    - transpose
    - optimize alpha version
    - edge cases
"""


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
        self.K_blk = K_blk
        self.M_blk = M_blk
        self.N_blk = N_blk
        self.M_reg = M_reg
        self.N_reg = N_reg

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

        @proc
        def gemm_notranspose_alpha(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            temp: f32[K, N]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[k, j] = B[k, j] * alpha[0]
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * temp[k, j]

        gemm_notranspose_alpha = rename(
            gemm_notranspose_alpha,
            f"{self.prefix}{gemm_notranspose_alpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_notranspose_alpha = self.specialize_gemm(
            gemm_notranspose_alpha, self.precision, ["A", "B", "C", "alpha", "temp"]
        )

        @proc
        def gemm_transa_noalpha(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            assert M == K
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * B[k, j]

        gemm_transa_noalpha = rename(
            gemm_transa_noalpha,
            f"{self.prefix}{gemm_transa_noalpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transa_noalpha = self.specialize_gemm(
            gemm_transa_noalpha, self.precision, ["A", "B", "C"]
        )

        @proc
        def gemm_transa_alpha(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            C: f32[M, N] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            assert M == K
            temp: f32[K, N]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[k, j] = B[k, j] * alpha[0]
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * temp[k, j]

        gemm_transa_alpha = rename(
            gemm_transa_alpha,
            f"{self.prefix}{gemm_transa_alpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transa_alpha = self.specialize_gemm(
            gemm_transa_alpha, self.precision, ["A", "B", "C", "alpha", "temp"]
        )

        @proc
        def gemm_transb_noalpha(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
        ):
            assert K == N
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * B[j, k]

        gemm_transb_noalpha = rename(
            gemm_transb_noalpha,
            f"{self.prefix}{gemm_transb_noalpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transb_noalpha = self.specialize_gemm(
            gemm_transb_noalpha, self.precision, ["A", "B", "C"]
        )

        @proc
        def gemm_transb_alpha(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
        ):
            assert N == K
            temp: f32[N, K]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[k, j] = B[k, j] * alpha[0]
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * temp[j, k]

        gemm_transb_alpha = rename(
            gemm_transb_alpha,
            f"{self.prefix}{gemm_transb_alpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transb_alpha = self.specialize_gemm(
            gemm_transb_alpha, self.precision, ["A", "B", "C", "alpha", "temp"]
        )

        @proc
        def gemm_transa_transb_noalpha(
            M: size,
            N: size,
            K: size,
            C: f32[K, K] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[N, K] @ DRAM,
        ):
            assert M == N == K
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * B[j, k]

        gemm_transa_transb_noalpha = rename(
            gemm_transa_transb_noalpha,
            f"{self.prefix}{gemm_transa_transb_noalpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transa_transb_noalpha = self.specialize_gemm(
            gemm_transa_transb_noalpha, self.precision, ["A", "B", "C"]
        )

        @proc
        def gemm_transa_transb_alpha(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            C: f32[K, K] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[N, K] @ DRAM,
        ):
            assert N == K == M
            temp: f32[N, K]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[k, j] = B[k, j] * alpha[0]
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * temp[j, k]

        gemm_transa_transb_alpha = rename(
            gemm_transa_transb_alpha,
            f"{self.prefix}{gemm_transa_transb_alpha.name()}_{N_blk}_{M_blk}_{K_blk}",
        )
        gemm_transa_transb_alpha = self.specialize_gemm(
            gemm_transa_transb_alpha, self.precision, ["A", "B", "C", "alpha", "temp"]
        )

        ### ALPHA AND BETA
        @proc
        def gemm_apply_scalar(M: size, N: size, scalar: f32[1], P: f32[M, N] @ DRAM):
            for i in seq(0, M):
                for j in seq(0, N):
                    P[i, j] = P[i, j] * scalar[0]

        gemm_apply_scalar = set_precision(gemm_apply_scalar, "scalar", self.precision)
        gemm_apply_scalar = set_precision(gemm_apply_scalar, "P", self.precision)

        @proc
        def gemm_apply_scalar_no_overwrite(
            M: size, N: size, scalar: f32[1], P: f32[M, N] @ DRAM, Q: f32[M, N] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    Q[i, j] = P[i, j] * scalar[0]

        gemm_apply_scalar_no_overwrite = set_precision(
            gemm_apply_scalar_no_overwrite, "scalar", self.precision
        )
        gemm_apply_scalar_no_overwrite = set_precision(
            gemm_apply_scalar_no_overwrite, "P", self.precision
        )
        gemm_apply_scalar_no_overwrite = set_precision(
            gemm_apply_scalar_no_overwrite, "Q", self.precision
        )

        ### Alpha and Beta scaling procedures
        apply_alpha = self.schedule_apply_scalar(
            gemm_apply_scalar_no_overwrite,
            machine,
            ["Q", "P"],
            f"{self.prefix}_apply_alpha_{N_blk}_{M_blk}_{K_blk}",
            False,
        )
        apply_beta = self.schedule_apply_scalar(
            gemm_apply_scalar,
            machine,
            ["P"],
            f"{self.prefix}_apply_beta_{N_blk}_{M_blk}_{K_blk}",
            True,
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
        gemm_scheduled_notranspose_alpha = self.schedule_gemm_notranspose_alpha(
            gemm_notranspose_alpha, gemm_apply_scalar_no_overwrite, apply_alpha
        )
        # gemm_scheduled_transpose_noalpha
        # gemm_scheduled_transpose_alpha

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

        @proc
        def exo_gemm_transa_noalpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert M == K
            gemm_transa_noalpha(M, N, K, C, A, B)

        exo_gemm_transa_noalpha_nobeta = self.specialize_gemm(
            exo_gemm_transa_noalpha_nobeta, self.precision
        )

        @proc
        def exo_gemm_transb_noalpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert N == K
            gemm_transb_noalpha(M, N, K, C, A, B)

        exo_gemm_transb_noalpha_nobeta = self.specialize_gemm(
            exo_gemm_transb_noalpha_nobeta, self.precision
        )

        @proc
        def exo_gemm_transa_transb_noalpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[K, K] @ DRAM,
        ):
            assert N == K == M
            gemm_transa_transb_noalpha(M, N, K, C, A, B)

        exo_gemm_transa_transb_noalpha_nobeta = self.specialize_gemm(
            exo_gemm_transa_transb_noalpha_nobeta, self.precision
        )

        # Alpha = 0, Beta = 0
        @proc
        def exo_gemm_alphazero_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            pass

        exo_gemm_alphazero_nobeta = self.specialize_gemm(
            exo_gemm_alphazero_nobeta, self.precision
        )

        # Alpha = 0, Beta != 0
        @proc
        def exo_gemm_alphazero_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            apply_beta(M, N, beta, C)

        exo_gemm_alphazero_beta = self.specialize_gemm(
            exo_gemm_alphazero_beta, self.precision
        )

        # Alpha != 1, Beta = 1
        @proc
        def exo_gemm_notranspose_alpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            gemm_scheduled_notranspose_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_notranspose_alpha_nobeta = self.specialize_gemm(
            exo_gemm_notranspose_alpha_nobeta, self.precision
        )

        @proc
        def exo_gemm_transa_alpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert M == K
            gemm_transa_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transa_alpha_nobeta = self.specialize_gemm(
            exo_gemm_transa_alpha_nobeta, self.precision
        )

        @proc
        def exo_gemm_transb_alpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert N == K
            gemm_transb_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transb_alpha_nobeta = self.specialize_gemm(
            exo_gemm_transb_alpha_nobeta, self.precision
        )

        @proc
        def exo_gemm_transa_transb_alpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[K, K] @ DRAM,
        ):
            assert N == K == M
            gemm_transa_transb_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transa_transb_alpha_nobeta = self.specialize_gemm(
            exo_gemm_transa_transb_alpha_nobeta, self.precision
        )

        # Alpha = Beta != 1
        @proc
        def exo_gemm_notranspose_alpha_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            apply_beta(M, N, beta, C)
            gemm_scheduled_notranspose_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_notranspose_alpha_beta = self.specialize_gemm(
            exo_gemm_notranspose_alpha_beta, self.precision
        )

        @proc
        def exo_gemm_transa_alpha_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert K == M
            apply_beta(M, N, beta, C)
            gemm_transa_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transa_alpha_beta = self.specialize_gemm(
            exo_gemm_transa_alpha_beta, self.precision
        )

        @proc
        def exo_gemm_transb_alpha_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[M, N] @ DRAM,
        ):
            assert N == K
            apply_beta(M, N, beta, C)
            gemm_transb_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transb_alpha_beta = self.specialize_gemm(
            exo_gemm_transb_alpha_beta, self.precision
        )

        @proc
        def exo_gemm_transa_transb_alpha_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[K, M] @ DRAM,
            B: f32[N, K] @ DRAM,
            C: f32[K, K] @ DRAM,
        ):
            assert N == K == M
            apply_beta(M, N, beta, C)
            gemm_transa_transb_alpha(M, N, K, alpha, C, A, B)

        exo_gemm_transa_transb_alpha_beta = self.specialize_gemm(
            exo_gemm_transa_transb_alpha_beta, self.precision
        )

        self.entry_points = [
            exo_gemm_notranspose_noalpha_nobeta,
            exo_gemm_alphazero_nobeta,
            exo_gemm_alphazero_beta,
            exo_gemm_notranspose_alpha_nobeta,
            exo_gemm_notranspose_alpha_beta,
            exo_gemm_transa_noalpha_nobeta,
            exo_gemm_transa_alpha_nobeta,
            exo_gemm_transa_alpha_beta,
            exo_gemm_transb_noalpha_nobeta,
            exo_gemm_transb_alpha_nobeta,
            exo_gemm_transb_alpha_beta,
            exo_gemm_transa_transb_noalpha_nobeta,
            exo_gemm_transa_transb_alpha_nobeta,
            exo_gemm_transa_transb_alpha_beta,
        ]

        if do_rename:
            for i in range(len(self.entry_points)):
                self.entry_points[i] = rename(
                    self.entry_points[i],
                    f"{self.entry_points[i].name()}_{N_blk}_{M_blk}_{K_blk}",
                )

    def schedule_gemm_notranspose_noalpha_compact(self, gemm_procedure: Procedure):
        gemm_scheduled = gemm_procedure
        i_loop = gemm_scheduled.find_loop("i")
        j_loop = gemm_scheduled.find_loop("j")
        k_loop = gemm_scheduled.find_loop("k")
        gemm_scheduled = reorder_loops(gemm_scheduled, i_loop)
        gemm_scheduled = reorder_loops(gemm_scheduled, i_loop)
        gemm_scheduled, _ = tile_loops(
            gemm_scheduled,
            [(j_loop, self.N_blk), (k_loop, self.K_blk), (i_loop, self.M_blk)],
        )
        ii_loop = gemm_scheduled.find_loop("ii")
        ji_loop = gemm_scheduled.find_loop("ji")
        ki_loop = gemm_scheduled.find_loop("ki")
        gemm_scheduled = reorder_loops(gemm_scheduled, ki_loop)
        gemm_scheduled, _ = tile_loops(
            gemm_scheduled, [(ji_loop, self.N_reg), (ii_loop, self.M_reg)]
        )
        gemm_scheduled = simplify(gemm_scheduled)
        gemm_scheduled = reorder_loops(gemm_scheduled, gemm_scheduled.find_loop("jii"))
        gemm_scheduled = simplify(
            auto_stage_mem(
                gemm_scheduled, gemm_scheduled.find("B[_]"), "B_reg_strip", n_lifts=4
            )
        )
        gemm_scheduled = lift_alloc(gemm_scheduled, "B_reg_strip", n_lifts=4)
        gemm_scheduled = set_memory(gemm_scheduled, "B_reg_strip: _", DRAM_STATIC)
        gemm_scheduled = simplify(
            auto_stage_mem(
                gemm_scheduled, gemm_scheduled.find("B[_]"), "B_strip", n_lifts=4
            )
        )
        gemm_scheduled = lift_alloc(gemm_scheduled, "B_strip", n_lifts=2)
        gemm_scheduled = set_memory(gemm_scheduled, "B_strip: _", DRAM_STATIC)
        gemm_scheduled = replace(
            gemm_scheduled,
            gemm_scheduled.find_loop("iii"),
            self.microkernel.base_microkernel,
        )
        call_c = gemm_scheduled.find(f"microkernel_{self.microkernel.this_id}(_)")
        gemm_scheduled = call_eqv(
            gemm_scheduled,
            call_c,
            self.microkernel.scheduled_microkernel,
        )
        gemm_scheduled = inline(gemm_scheduled, call_c)
        gemm_scheduled = inline_window(gemm_scheduled, "C = C[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "A = A[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "B = B_reg_strip[_]")
        gemm_scheduled = simplify(gemm_scheduled)

        return gemm_scheduled

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
            f"B[{self.microkernel.K_blk}*ko:{self.microkernel.K_blk}*ko+{self.microkernel.K_blk}, {self.gebp.N_blk}*jo:{self.gebp.N_blk}*jo+{self.gebp.N_blk}]",
            "B_strip",
        )
        gemm_scheduled = simplify(gemm_scheduled)

        gemm_scheduled = replace(gemm_scheduled, "for i in _:_ #0", self.gepp.gepp_base)
        call_c = gemm_scheduled.find(f"gepp_base_{self.gepp.this_id}(_)")
        gemm_scheduled = call_eqv(
            gemm_scheduled,
            call_c,
            self.gepp.gepp_scheduled,
        )
        gemm_scheduled = inline(gemm_scheduled, call_c)
        gemm_scheduled = inline_window(gemm_scheduled, "C = C[_]")
        gemm_scheduled = inline_window(gemm_scheduled, f"A = A[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "B = B_strip[_]")
        gemm_scheduled = simplify(gemm_scheduled)

        while True:
            try:
                gemm_scheduled = lift_alloc(gemm_scheduled, "B_reg_strip:_")
            except:
                break
        while True:
            try:
                gemm_scheduled = lift_alloc(gemm_scheduled, "B_strip:_")
            except:
                break
        gemm_scheduled = set_memory(gemm_scheduled, "B_strip:_", DRAM_STATIC)
        gemm_scheduled = set_memory(gemm_scheduled, "B_reg_strip:_", DRAM_STATIC)

        return gemm_scheduled

    def schedule_gemm_notranspose_alpha(
        self,
        gemm_procedure: Procedure,
        apply_alpha_base: Procedure,
        apply_alpha_scheduled: Procedure,
    ):

        gemm_scheduled = divide_loop(
            gemm_procedure,
            "k #1",
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
            gemm_scheduled, "j #1", self.gebp.N_blk, ["jo", "ji"], tail="cut"
        )
        gemm_scheduled = autofission(
            gemm_scheduled, gemm_scheduled.find("for jo in _:_ #0ß").after(), n_lifts=2
        )
        gemm_scheduled = reorder_loops(gemm_scheduled, "i jo")
        gemm_scheduled = reorder_loops(gemm_scheduled, "ko jo")

        gemm_scheduled = replace(gemm_scheduled, "for i in _:_ #0", self.gepp.gepp_base)
        gemm_scheduled = call_eqv(
            gemm_scheduled,
            f"gepp_base_{self.gepp.this_id}(_)",
            self.gepp.gepp_scheduled,
        )

        gemm_scheduled = replace(
            gemm_scheduled, "for j in _:_ #0", reorder_loops(apply_alpha_base, "i j")
        )
        gemm_scheduled = call_eqv(
            gemm_scheduled, "gemm_apply_scalar_no_overwrite(_)", apply_alpha_scheduled
        )

        gemm_scheduled = simplify(gemm_scheduled)

        return gemm_scheduled

    def bind(self, proc, buffer, reg, machine):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, machine.vec_width, "ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg} = _").after())
        return proc

    def stage(self, proc, buffer, reg, machine):
        proc = stage_mem(
            proc, f"{buffer}[_] = _", f"{buffer}[i, ji + {machine.vec_width}*jo]", reg
        )
        proc = expand_dim(proc, reg, machine.vec_width, f"ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc

    def schedule_apply_scalar(
        self,
        proc: Procedure,
        machine: "MachineParameters",
        buffer_names: list,
        name: str,
        apply_hack: bool,
    ):

        proc = rename(proc, name)
        for buffer in buffer_names + ["scalar"]:
            proc = set_precision(proc, buffer, self.precision)

        proc = divide_loop(
            proc, "j", machine.vec_width, ["jo", "ji"], tail="cut_and_guard"
        )
        proc = self.bind(proc, "scalar[_]", "scalar_vec", machine)
        if len(buffer_names) > 1:
            proc = self.bind(
                proc, f"{buffer_names[1]}[_]", f"{buffer_names[1]}_vec", machine
            )
            proc = set_precision(proc, f"{buffer_names[1]}_vec", self.precision)
        proc = self.stage(proc, f"{buffer_names[0]}", f"{buffer_names[0]}_vec", machine)

        if apply_hack:
            proc = fission(proc, proc.find(f"{buffer_names[0]}_vec[_] = _ #1").after())

        for buffer_name in buffer_names:
            proc = set_memory(proc, f"{buffer_name}_vec", machine.mem_type)
        proc = set_memory(proc, "scalar_vec", machine.mem_type)
        proc = set_precision(proc, "scalar_vec", self.precision)

        if self.precision == "f32":
            instr_lst = [
                machine.load_instr_f32,
                machine.broadcast_instr_f32,
                machine.reg_copy_instr_f32,
                machine.mul_instr_f32,
                machine.store_instr_f32,
            ]
        else:
            instr_lst = [
                machine.load_instr_f64,
                machine.broadcast_instr_f64,
                machine.reg_copy_instr_f64,
                machine.mul_instr_f64,
                machine.store_instr_f64,
            ]

        if apply_hack:
            proc = self.bind(proc, "P_vec[_]", "P_vec2", machine)
            proc = set_memory(proc, "P_vec2", machine.mem_type)
            proc = set_precision(proc, "P_vec2", self.precision)

        for instr in instr_lst:
            proc = replace_all(proc, instr)

        if self.main:
            proc = rename(proc, proc.name() + "_main")

        return simplify(proc)

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

"""
# sgemm_main = GEMM(C.Machine, "f32", k_blk, m_blk, n_blk, m_reg, n_reg)

square_blk_sizes = [2**i for i in range(5, 9)]
sgemm_square_kernels = [
    GEMM(C.Machine, "f32", blk, blk, blk, m_reg, n_reg, True, False)
    for blk in square_blk_sizes
]

n_blk_sizes = [2**i for i in range(9, 14)]
m_blk = 256
k_blk = 512
sgemm_large_kernels = [
    GEMM(C.Machine, "f32", k_blk, m_blk, _n_blk, m_reg, n_reg, True, False)
    for _n_blk in n_blk_sizes
]

exo_sgemm_notranspose_noalpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[0]
exo_sgemm_alphazero_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[1]
exo_sgemm_alphazero_beta_32_32_32 = sgemm_square_kernels[0].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[3]
exo_sgemm_notranspose_alpha_beta_32_32_32 = sgemm_square_kernels[0].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[5]
exo_sgemm_transa_alpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[6]
exo_sgemm_transa_alpha_beta_32_32_32 = sgemm_square_kernels[0].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[8]
exo_sgemm_transb_alpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[9]
exo_sgemm_transb_alpha_beta_32_32_32 = sgemm_square_kernels[0].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[
    11
]
exo_sgemm_transa_transb_alpha_nobeta_32_32_32 = sgemm_square_kernels[0].entry_points[12]
exo_sgemm_transa_transb_alpha_beta_32_32_32 = sgemm_square_kernels[0].entry_points[13]

exo_sgemm_notranspose_noalpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[0]
exo_sgemm_alphazero_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[1]
exo_sgemm_alphazero_beta_64_64_64 = sgemm_square_kernels[1].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[3]
exo_sgemm_notranspose_alpha_beta_64_64_64 = sgemm_square_kernels[1].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[5]
exo_sgemm_transa_alpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[6]
exo_sgemm_transa_alpha_beta_64_64_64 = sgemm_square_kernels[1].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[8]
exo_sgemm_transb_alpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[9]
exo_sgemm_transb_alpha_beta_64_64_64 = sgemm_square_kernels[1].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[
    11
]
exo_sgemm_transa_transb_alpha_nobeta_64_64_64 = sgemm_square_kernels[1].entry_points[12]
exo_sgemm_transa_transb_alpha_beta_64_64_64 = sgemm_square_kernels[1].entry_points[13]

exo_sgemm_notranspose_noalpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[0]
exo_sgemm_alphazero_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[1]
exo_sgemm_alphazero_beta_128_128_128 = sgemm_square_kernels[2].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[3]
exo_sgemm_notranspose_alpha_beta_128_128_128 = sgemm_square_kernels[2].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[5]
exo_sgemm_transa_alpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[6]
exo_sgemm_transa_alpha_beta_128_128_128 = sgemm_square_kernels[2].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[8]
exo_sgemm_transb_alpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[9]
exo_sgemm_transb_alpha_beta_128_128_128 = sgemm_square_kernels[2].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_128_128_128 = sgemm_square_kernels[
    2
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_128_128_128 = sgemm_square_kernels[2].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_128_128_128 = sgemm_square_kernels[2].entry_points[
    13
]

exo_sgemm_notranspose_noalpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[0]
exo_sgemm_alphazero_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[1]
exo_sgemm_alphazero_beta_256_256_256 = sgemm_square_kernels[3].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[3]
exo_sgemm_notranspose_alpha_beta_256_256_256 = sgemm_square_kernels[3].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[5]
exo_sgemm_transa_alpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[6]
exo_sgemm_transa_alpha_beta_256_256_256 = sgemm_square_kernels[3].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[8]
exo_sgemm_transb_alpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[9]
exo_sgemm_transb_alpha_beta_256_256_256 = sgemm_square_kernels[3].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_256_256_256 = sgemm_square_kernels[
    3
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_256_256_256 = sgemm_square_kernels[3].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_256_256_256 = sgemm_square_kernels[3].entry_points[
    13
]

exo_sgemm_notranspose_noalpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[0]
exo_sgemm_alphazero_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[1]
exo_sgemm_alphazero_beta_512_256_512 = sgemm_large_kernels[0].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[3]
exo_sgemm_notranspose_alpha_beta_512_256_512 = sgemm_large_kernels[0].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[5]
exo_sgemm_transa_alpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[6]
exo_sgemm_transa_alpha_beta_512_256_512 = sgemm_large_kernels[0].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[8]
exo_sgemm_transb_alpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[9]
exo_sgemm_transb_alpha_beta_512_256_512 = sgemm_large_kernels[0].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_512_256_512 = sgemm_large_kernels[
    0
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_512_256_512 = sgemm_large_kernels[0].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_512_256_512 = sgemm_large_kernels[0].entry_points[13]

exo_sgemm_notranspose_noalpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[0]
exo_sgemm_alphazero_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[1]
exo_sgemm_alphazero_beta_1024_256_512 = sgemm_large_kernels[1].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[3]
exo_sgemm_notranspose_alpha_beta_1024_256_512 = sgemm_large_kernels[1].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[5]
exo_sgemm_transa_alpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[6]
exo_sgemm_transa_alpha_beta_1024_256_512 = sgemm_large_kernels[1].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[8]
exo_sgemm_transb_alpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[9]
exo_sgemm_transb_alpha_beta_1024_256_512 = sgemm_large_kernels[1].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_1024_256_512 = sgemm_large_kernels[
    1
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_1024_256_512 = sgemm_large_kernels[1].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_1024_256_512 = sgemm_large_kernels[1].entry_points[
    13
]

exo_sgemm_notranspose_noalpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[0]
exo_sgemm_alphazero_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[1]
exo_sgemm_alphazero_beta_2048_256_512 = sgemm_large_kernels[2].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[3]
exo_sgemm_notranspose_alpha_beta_2048_256_512 = sgemm_large_kernels[2].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[5]
exo_sgemm_transa_alpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[6]
exo_sgemm_transa_alpha_beta_2048_256_512 = sgemm_large_kernels[2].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[8]
exo_sgemm_transb_alpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[9]
exo_sgemm_transb_alpha_beta_2048_256_512 = sgemm_large_kernels[2].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_2048_256_512 = sgemm_large_kernels[
    2
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_2048_256_512 = sgemm_large_kernels[2].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_2048_256_512 = sgemm_large_kernels[2].entry_points[
    13
]

exo_sgemm_notranspose_noalpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[0]
exo_sgemm_alphazero_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[1]
exo_sgemm_alphazero_beta_4096_256_512 = sgemm_large_kernels[3].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[3]
exo_sgemm_notranspose_alpha_beta_4096_256_512 = sgemm_large_kernels[3].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[5]
exo_sgemm_transa_alpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[6]
exo_sgemm_transa_alpha_beta_4096_256_512 = sgemm_large_kernels[3].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[8]
exo_sgemm_transb_alpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[9]
exo_sgemm_transb_alpha_beta_4096_256_512 = sgemm_large_kernels[3].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_4096_256_512 = sgemm_large_kernels[
    3
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_4096_256_512 = sgemm_large_kernels[3].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_4096_256_512 = sgemm_large_kernels[3].entry_points[
    13
]

exo_sgemm_notranspose_noalpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[0]
exo_sgemm_alphazero_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[1]
exo_sgemm_alphazero_beta_8192_256_512 = sgemm_large_kernels[4].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[3]
exo_sgemm_notranspose_alpha_beta_8192_256_512 = sgemm_large_kernels[4].entry_points[4]
exo_sgemm_transa_noalpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[5]
exo_sgemm_transa_alpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[6]
exo_sgemm_transa_alpha_beta_8192_256_512 = sgemm_large_kernels[4].entry_points[7]
exo_sgemm_transb_noalpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[8]
exo_sgemm_transb_alpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[9]
exo_sgemm_transb_alpha_beta_8192_256_512 = sgemm_large_kernels[4].entry_points[10]
exo_sgemm_transa_transb_noalpha_nobeta_8192_256_512 = sgemm_large_kernels[
    4
].entry_points[11]
exo_sgemm_transa_transb_alpha_nobeta_8192_256_512 = sgemm_large_kernels[4].entry_points[
    12
]
exo_sgemm_transa_transb_alpha_beta_8192_256_512 = sgemm_large_kernels[4].entry_points[
    13
]

sgemm_square_entry_points = []
for kernel in sgemm_square_kernels:
    sgemm_square_entry_points.extend(kernel.entry_points)

sgemm_large_entry_points = []
for kernel in sgemm_large_kernels:
    sgemm_large_entry_points.extend(kernel.entry_points)

sgemm_entry_points = [p.name() for p in sgemm_square_entry_points] + [
    p.name() for p in sgemm_large_entry_points
]

#################################################
# Generate f64 kernels
#################################################

C.Machine.vec_width //= 2

dgemm_main = GEMM(C.Machine, "f64", k_blk, m_blk, n_blk, m_reg, n_reg // 2)
dgemm_backup_kernels = [GEMM(C.Machine, 'f64', blk, blk, blk, m_reg, n_reg//2, True, False) for blk in blk_sizes] # Use these if problem size is too small for the main block size

exo_dgemm_notranspose_noalpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[0]
exo_dgemm_alphazero_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[1]
exo_dgemm_alphazero_beta_32_32 = dgemm_backup_kernels[0].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[3]
exo_dgemm_notranspose_alpha_beta_32_32 = dgemm_backup_kernels[0].entry_points[4]
exo_dgemm_transa_noalpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[5]
exo_dgemm_transa_alpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[6]
exo_dgemm_transa_alpha_beta_32_32 = dgemm_backup_kernels[0].entry_points[7]
exo_dgemm_transb_noalpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[8]
exo_dgemm_transb_alpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[9]
exo_dgemm_transb_alpha_beta_32_32 = dgemm_backup_kernels[0].entry_points[10]
exo_dgemm_transa_transb_noalpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[11]
exo_dgemm_transa_transb_alpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[12]
exo_dgemm_transa_transb_alpha_beta_32_32 = dgemm_backup_kernels[0].entry_points[13]

exo_dgemm_notranspose_noalpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[0]
exo_dgemm_alphazero_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[1]
exo_dgemm_alphazero_beta_64_64_64 = dgemm_backup_kernels[1].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[3]
exo_dgemm_notranspose_alpha_beta_64_64_64 = dgemm_backup_kernels[1].entry_points[4]
exo_dgemm_transa_noalpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[5]
exo_dgemm_transa_alpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[6]
exo_dgemm_transa_alpha_beta_64_64_64 = dgemm_backup_kernels[1].entry_points[7]
exo_dgemm_transb_noalpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[8]
exo_dgemm_transb_alpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[9]
exo_dgemm_transb_alpha_beta_64_64_64 = dgemm_backup_kernels[1].entry_points[10]
exo_dgemm_transa_transb_noalpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[11]
exo_dgemm_transa_transb_alpha_nobeta_64_64_64 = dgemm_backup_kernels[1].entry_points[12]
exo_dgemm_transa_transb_alpha_beta_64_64_64 = dgemm_backup_kernels[1].entry_points[13]

exo_dgemm_notranspose_noalpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[0]
exo_dgemm_alphazero_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[1]
exo_dgemm_alphazero_beta_128_128_128 = dgemm_backup_kernels[2].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[3]
exo_dgemm_notranspose_alpha_beta_128_128_128 = dgemm_backup_kernels[2].entry_points[4]
exo_dgemm_transa_noalpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[5]
exo_dgemm_transa_alpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[6]
exo_dgemm_transa_alpha_beta_128_128_128 = dgemm_backup_kernels[2].entry_points[7]
exo_dgemm_transb_noalpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[8]
exo_dgemm_transb_alpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[9]
exo_dgemm_transb_alpha_beta_128_128_128 = dgemm_backup_kernels[2].entry_points[10]
exo_dgemm_transa_transb_noalpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[
    11
]
exo_dgemm_transa_transb_alpha_nobeta_128_128_128 = dgemm_backup_kernels[2].entry_points[12]
exo_dgemm_transa_transb_alpha_beta_128_128_128 = dgemm_backup_kernels[2].entry_points[13]

exo_dgemm_notranspose_noalpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[0]
exo_dgemm_alphazero_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[1]
exo_dgemm_alphazero_beta_256_256_256 = dgemm_backup_kernels[3].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[3]
exo_dgemm_notranspose_alpha_beta_256_256_256 = dgemm_backup_kernels[3].entry_points[4]
exo_dgemm_transa_noalpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[5]
exo_dgemm_transa_alpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[6]
exo_dgemm_transa_alpha_beta_256_256_256 = dgemm_backup_kernels[3].entry_points[7]
exo_dgemm_transb_noalpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[8]
exo_dgemm_transb_alpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[9]
exo_dgemm_transb_alpha_beta_256_256_256 = dgemm_backup_kernels[3].entry_points[10]
exo_dgemm_transa_transb_noalpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[
    11
]
exo_dgemm_transa_transb_alpha_nobeta_256_256_256 = dgemm_backup_kernels[3].entry_points[12]
exo_dgemm_transa_transb_alpha_beta_256_256_256 = dgemm_backup_kernels[3].entry_points[13]
exo_dgemm_notranspose_noalpha_nobeta_main = dgemm_main.entry_points[0]
exo_dgemm_alphazero_nobeta_main = dgemm_main.entry_points[1]
exo_dgemm_alphazero_beta_main = dgemm_main.entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_main = dgemm_main.entry_points[3]
exo_dgemm_notranspose_alpha_beta_main = dgemm_main.entry_points[4]
exo_dgemm_transa_noalpha_nobeta_main = dgemm_main.entry_points[5]
exo_dgemm_transa_alpha_nobeta_main = dgemm_main.entry_points[6]
exo_dgemm_transa_alpha_beta_main = dgemm_main.entry_points[7]
exo_dgemm_transb_noalpha_nobeta_main = dgemm_main.entry_points[8]
exo_dgemm_transb_alpha_nobeta_main = dgemm_main.entry_points[9]
exo_dgemm_transb_alpha_beta_main = dgemm_main.entry_points[10]
exo_dgemm_transa_transb_noalpha_nobeta_main = dgemm_main.entry_points[11]
exo_dgemm_transa_transb_alpha_nobeta_main = dgemm_main.entry_points[12]
exo_dgemm_transa_transb_alpha_beta_main = dgemm_main.entry_points[13]


# dgemm_backup_entry_points = []
# for kernel in dgemm_backup_kernels:
#    dgemm_backup_entry_points.extend(kernel.entry_points)
dgemm_entry_points = [
    p.name() for p in dgemm_main.entry_points
]  # + [p.name() for p in dgemm_backup_entry_points]
"""

exo_sgemm_notranspose_noalpha_nobeta_48_48_48 = GEMM(
    C.Machine, "f32", 48, 48, 48, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_96_96_96 = GEMM(
    C.Machine, "f32", 96, 96, 96, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_192_192_192 = GEMM(
    C.Machine, "f32", 192, 192, 192, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_384_384_384 = GEMM(
    C.Machine, "f32", 384, 384, 384, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_528_240_528 = GEMM(
    C.Machine, "f32", 528, 240, 528, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_1056_240_528 = GEMM(
    C.Machine, "f32", 528, 240, 1056, m_reg, n_reg, True, False
).entry_points[0]
exo_sgemm_notranspose_noalpha_nobeta_2112_240_528 = GEMM(
    C.Machine, "f32", 528, 240, 2112, m_reg, n_reg, True, False
).entry_points[0]

benchmark_points = [
    "exo_sgemm_notranspose_noalpha_nobeta_48_48_48",
    "exo_sgemm_notranspose_noalpha_nobeta_96_96_96",
    "exo_sgemm_notranspose_noalpha_nobeta_192_192_192",
    "exo_sgemm_notranspose_noalpha_nobeta_384_384_384",
    "exo_sgemm_notranspose_noalpha_nobeta_528_240_528",
    "exo_sgemm_notranspose_noalpha_nobeta_1056_240_528",
    "exo_sgemm_notranspose_noalpha_nobeta_2112_240_528",
]

# __all__ = sgemm_entry_points + dgemm_entry_points + benchmark_points
__all__ = benchmark_points
