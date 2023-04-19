from __future__ import annotations
from os import abort

import sys
import getopt
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEBP_kernel, Microkernel
from format_options import *
from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)

import exo_blas_config as C


class SYRK:
    """
    TODO: Add Beta and Alpha
    """

    def __init__(
        self,
        machine: "MachineParameters",
        precision: str,
        K_blk: int,
        M_blk: int,
        M_blk_small: int,
        M_r: int,
        N_r: int,
        e_reg: int,
    ):

        # Precision
        self.precision = precision
        self.prefix = "s" if precision == "f32" else "d"
        # print(M_r, N_r)

        # Generate kernels
        self.microkernel = Microkernel(machine, M_r, N_r, K_blk, self.precision)
        self.gebp_kernel = GEBP_kernel(self.microkernel, M_blk, M_blk, self.precision)

        # Blocking dimensions
        self.K_blk = K_blk
        self.M_blk = M_blk
        self.M_blk_small = M_blk_small
        self.e_reg = e_reg

        # Machine
        self.machine = machine

        ### SYRK procedures
        @proc
        def syrk_lower_notranspose_noalpha(
            N: size,
            K: size,
            A1: f32[N, K] @ DRAM,
            A2: f32[K, N] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            # C = A*A**T + C
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1

            for i in seq(0, N):
                for j in seq(0, i + 1):
                    for k in seq(0, K):
                        C[i, j] += A1[i, k] * A2[k, j]

        syrk_lower_notranspose_noalpha = self.specialize_syrk(
            syrk_lower_notranspose_noalpha, self.precision, ["A1", "A2", "C"]
        )
        syrk_lower_notranspose_noalpha = rename(
            syrk_lower_notranspose_noalpha,
            f"{self.prefix}{syrk_lower_notranspose_noalpha.name()}",
        )

        @proc
        def syrk_lower_transpose_noalpha(
            N: size,
            K: size,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            # C = A**T*A + C
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1
            assert N == K
            for i in seq(0, N):
                for j in seq(0, i + 1):
                    for k in seq(0, K):
                        C[i, j] += A1[k, i] * A2[k, j]

        syrk_lower_transpose_noalpha = self.specialize_syrk(
            syrk_lower_transpose_noalpha, self.precision, ["A1", "A2", "C"]
        )
        syrk_lower_transpose_noalpha = rename(
            syrk_lower_transpose_noalpha,
            f"{self.prefix}{syrk_lower_transpose_noalpha.name()}",
        )

        @proc
        def syrk_lower_notranspose_alpha(
            N: size,
            K: size,
            A1: f32[N, K] @ DRAM,
            alpha: f32[1],
            A2: f32[N, K] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):

            for i in seq(0, N):
                for j in seq(0, i + 1):
                    temp: f32[1]
                    temp[0] = 0.0
                    for k in seq(0, K):
                        temp[0] += A1[i, k] * A2[j, k]
                    C[i, j] += alpha[0] * temp[0]

        syrk_lower_notranspose_alpha = self.specialize_syrk(
            syrk_lower_notranspose_alpha,
            self.precision,
            ["A1", "A2", "C", "alpha", "temp"],
        )
        syrk_lower_notranspose_alpha = rename(
            syrk_lower_notranspose_alpha,
            f"{self.prefix}{syrk_lower_notranspose_alpha.name()}",
        )

        @proc
        def syrk_lower_transpose_alpha(
            N: size,
            K: size,
            A1: f32[K, N] @ DRAM,
            alpha: f32[1],
            A2: f32[K, N] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert N == K
            temp: f32[K, N]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[j, k] = A1[j, k] * alpha[0]
            for i in seq(0, N):
                for j in seq(0, i + 1):
                    for k in seq(0, K):
                        C[i, j] += temp[k, i] * A2[k, j]

        syrk_lower_transpose_alpha = self.specialize_syrk(
            syrk_lower_transpose_alpha,
            self.precision,
            ["A1", "A2", "C", "alpha", "temp"],
        )
        syrk_lower_transpose_alpha = rename(
            syrk_lower_transpose_alpha,
            f"{self.prefix}{syrk_lower_transpose_alpha.name()}",
        )

        @proc
        def syrk_upper_notranspose_noalpha(
            N: size,
            K: size,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            for j in seq(0, N):
                for k in seq(0, K):
                    for i in seq(0, j + 1):
                        C[i, j] += A1[i, k] * A2[j, k]

        syrk_upper_notranspose_noalpha = self.specialize_syrk(
            syrk_upper_notranspose_noalpha, self.precision, ["A1", "A2", "C"]
        )
        syrk_upper_notranspose_noalpha = rename(
            syrk_upper_notranspose_noalpha,
            f"{self.prefix}{syrk_upper_notranspose_noalpha.name()}",
        )

        @proc
        def syrk_upper_transpose_noalpha(
            N: size,
            K: size,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert K == N
            for j in seq(0, N):
                for k in seq(0, K):
                    for i in seq(0, j + 1):
                        C[i, j] += A1[k, i] * A2[k, j]

        syrk_upper_transpose_noalpha = self.specialize_syrk(
            syrk_upper_transpose_noalpha, self.precision, ["A1", "A2", "C"]
        )
        syrk_upper_transpose_noalpha = rename(
            syrk_upper_transpose_noalpha,
            f"{self.prefix}{syrk_upper_transpose_noalpha.name()}",
        )

        @proc
        def syrk_upper_notranspose_alpha(
            N: size,
            K: size,
            A1: f32[N, K] @ DRAM,
            alpha: f32[1] @ DRAM,
            A2: f32[N, K] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            for j in seq(0, N):
                for k in seq(0, K):
                    for i in seq(0, j + 1):
                        C[i, j] += alpha[0] * A1[i, k] * A2[j, k]

        syrk_upper_notranspose_alpha = self.specialize_syrk(
            syrk_upper_notranspose_alpha, self.precision, ["A1", "A2", "C", "alpha"]
        )
        syrk_upper_notranspose_alpha = rename(
            syrk_upper_notranspose_alpha,
            f"{self.prefix}{syrk_upper_notranspose_alpha.name()}",
        )

        @proc
        def syrk_upper_transpose_alpha(
            N: size,
            K: size,
            A1: f32[K, N] @ DRAM,
            alpha: f32[1] @ DRAM,
            A2: f32[K, N] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert K == N
            for j in seq(0, N):
                for k in seq(0, K):
                    for i in seq(0, j + 1):
                        C[i, j] += A1[k, i] * A2[k, j] * alpha[0]

        syrk_upper_transpose_alpha = self.specialize_syrk(
            syrk_upper_transpose_alpha, self.precision, ["A1", "A2", "C", "alpha"]
        )
        syrk_upper_transpose_alpha = rename(
            syrk_upper_transpose_alpha,
            f"{self.prefix}{syrk_upper_transpose_alpha.name()}",
        )

        ### Diagonal handlers
        @proc
        def diag_handler_lower_notranspose(
            N: size,
            K: size,
            A1: [f32][N, K] @ DRAM,
            A2: [f32][K, N] @ DRAM,
            C: [f32][N, N] @ DRAM,
        ):
            # C = A*A**T + C
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1

            for i in seq(0, N):
                for j in seq(0, i):
                    for k in seq(0, K):
                        C[i, j] += A1[i, k] * A2[k, j]

        diag_handler_lower_notranspose = set_precision(
            diag_handler_lower_notranspose, "A1", self.precision
        )
        diag_handler_lower_notranspose = set_precision(
            diag_handler_lower_notranspose, "A2", self.precision
        )
        diag_handler_lower_notranspose = set_precision(
            diag_handler_lower_notranspose, "C", self.precision
        )
        diag_handler_lower_notranspose = rename(
            diag_handler_lower_notranspose,
            f"{self.prefix}_{diag_handler_lower_notranspose.name()}",
        )

        ### Scaling procedures
        @proc
        def syrk_apply_scalar_lower(
            M: size, N: size, scalar: f32[1], P: f32[M, M] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, i + 1):
                    P[i, j] = P[i, j] * scalar[0]

        syrk_apply_scalar_lower = set_precision(
            syrk_apply_scalar_lower, "scalar", self.precision
        )
        syrk_apply_scalar_lower = set_precision(
            syrk_apply_scalar_lower, "P", self.precision
        )

        @proc
        def syrk_apply_scalar_upper(
            M: size, N: size, scalar: f32[1], P: f32[M, M] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, M - i):
                    P[i, j] = P[i, j] * scalar[0]

        syrk_apply_scalar_upper = set_precision(
            syrk_apply_scalar_upper, "scalar", self.precision
        )
        syrk_apply_scalar_upper = set_precision(
            syrk_apply_scalar_upper, "P", self.precision
        )

        @proc
        def syrk_apply_scalar_lower_no_overwrite(
            M: size, N: size, scalar: f32[1], P: f32[M, N] @ DRAM, Q: f32[M, N] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    Q[i, j] = P[i, j] * scalar[0]

        syrk_apply_scalar_lower_no_overwrite = set_precision(
            syrk_apply_scalar_lower_no_overwrite, "scalar", self.precision
        )
        syrk_apply_scalar_lower_no_overwrite = set_precision(
            syrk_apply_scalar_lower_no_overwrite, "P", self.precision
        )
        syrk_apply_scalar_lower_no_overwrite = set_precision(
            syrk_apply_scalar_lower_no_overwrite, "Q", self.precision
        )

        ### Alpha and Beta scaling procedures
        # TODO: fix a lot here
        apply_alpha = self.schedule_apply_scalar(
            syrk_apply_scalar_lower_no_overwrite,
            machine,
            ["Q", "P"],
            f"{self.prefix}_apply_alpha_{M_blk}_{K_blk}",
            False,
        )
        apply_beta_lower = self.schedule_apply_scalar(
            syrk_apply_scalar_lower,
            machine,
            ["P"],
            f"{self.prefix}_apply_beta_{M_blk}_{K_blk}",
            True,
        )
        apply_beta_upper = rename(
            syrk_apply_scalar_upper, self.prefix + syrk_apply_scalar_upper.name()
        )

        ### Create scheduled procedures
        (
            self.gepp_syrk_scheduled_lower_notranspose,
            self.gepp_syrk_base_lower_notranspose,
        ) = self.generate_syrk_gepp_lower_notranspose_noalpha(
            syrk_lower_notranspose_noalpha, diag_handler_lower_notranspose
        )
        syrk_scheduled_lower_notranspose_noalpha = (
            self.schedule_syrk_lower_notranspose_noalpha(syrk_lower_notranspose_noalpha)
        )

        ### Entry points
        @proc
        def exo_syrk_lower_notranspose_noalpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            syrk_scheduled_lower_notranspose_noalpha(N, K, A1, A2, C)

        exo_syrk_lower_notranspose_noalpha_nobeta = self.specialize_syrk(
            exo_syrk_lower_notranspose_noalpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_notranspose_alpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            syrk_lower_notranspose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_lower_notranspose_alpha_nobeta = self.specialize_syrk(
            exo_syrk_lower_notranspose_alpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_notranspose_alpha_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            apply_beta_lower(N, N, beta, C)
            syrk_lower_notranspose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_lower_notranspose_alpha_beta = self.specialize_syrk(
            exo_syrk_lower_notranspose_alpha_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_transpose_noalpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert K == N
            syrk_lower_transpose_noalpha(N, K, A1, A2, C)

        exo_syrk_lower_transpose_noalpha_nobeta = self.specialize_syrk(
            exo_syrk_lower_transpose_noalpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_transpose_alpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert K == N
            syrk_lower_transpose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_lower_transpose_alpha_nobeta = self.specialize_syrk(
            exo_syrk_lower_transpose_alpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_transpose_alpha_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert K == N
            apply_beta_lower(N, N, beta, C)
            syrk_lower_transpose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_lower_transpose_alpha_beta = self.specialize_syrk(
            exo_syrk_lower_transpose_alpha_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_lower_alphazero_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            apply_beta_lower(N, N, beta, C)

        exo_syrk_lower_alphazero_beta = self.specialize_syrk(
            exo_syrk_lower_alphazero_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_notranspose_noalpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            syrk_upper_notranspose_noalpha(N, K, A1, A2, C)

        exo_syrk_upper_notranspose_noalpha_nobeta = self.specialize_syrk(
            exo_syrk_upper_notranspose_noalpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_notranspose_alpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            syrk_upper_notranspose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_upper_notranspose_alpha_nobeta = self.specialize_syrk(
            exo_syrk_upper_notranspose_alpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_notranspose_alpha_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            apply_beta_upper(N, N, beta, C)
            syrk_upper_notranspose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_upper_notranspose_alpha_beta = self.specialize_syrk(
            exo_syrk_upper_notranspose_alpha_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_transpose_noalpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert N == K
            syrk_upper_transpose_noalpha(N, K, A1, A2, C)

        exo_syrk_upper_transpose_noalpha_nobeta = self.specialize_syrk(
            exo_syrk_upper_transpose_noalpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_transpose_alpha_nobeta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert N == K
            syrk_upper_transpose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_upper_transpose_alpha_nobeta = self.specialize_syrk(
            exo_syrk_upper_transpose_alpha_nobeta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_transpose_alpha_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[K, N] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            assert N == K
            apply_beta_upper(N, N, beta, C)
            syrk_upper_transpose_alpha(N, K, A1, alpha, A2, C)

        exo_syrk_upper_transpose_alpha_beta = self.specialize_syrk(
            exo_syrk_upper_transpose_alpha_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        @proc
        def exo_syrk_upper_alphazero_beta(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[N, K] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM,
        ):
            apply_beta_upper(N, N, beta, C)

        exo_syrk_upper_alphazero_beta = self.specialize_syrk(
            exo_syrk_upper_alphazero_beta,
            self.precision,
            ["A1", "A2", "C", "alpha", "beta"],
        )

        self.entry_points = [
            exo_syrk_lower_notranspose_noalpha_nobeta,
            exo_syrk_lower_notranspose_alpha_nobeta,
            exo_syrk_lower_notranspose_alpha_beta,
            exo_syrk_lower_transpose_noalpha_nobeta,
            exo_syrk_lower_transpose_alpha_nobeta,
            exo_syrk_lower_transpose_alpha_beta,
            exo_syrk_upper_notranspose_noalpha_nobeta,
            exo_syrk_upper_notranspose_alpha_nobeta,
            exo_syrk_upper_notranspose_alpha_beta,
            exo_syrk_upper_transpose_noalpha_nobeta,
            exo_syrk_upper_transpose_alpha_nobeta,
            exo_syrk_upper_transpose_alpha_beta,
            exo_syrk_lower_alphazero_beta,
            exo_syrk_upper_alphazero_beta,
        ]

    def generate_syrk_gepp_base(self, syrk_win: Procedure):
        gepp_syrk_base = rename(syrk_win, "gepp_syrk_base")
        gepp_syrk_base = gepp_syrk_base.partial_eval(K=self.microkernel.K_blk)
        return gepp_syrk_base

    def generate_syrk_gepp_lower_notranspose_noalpha(
        self, syrk: Procedure, diag_handler: Procedure
    ):

        # assert(self.M_blk >= 128) # Temporary

        syrk = rename(syrk, "syrk_win")
        syrk = set_window(syrk, "A1", True)
        syrk = set_window(syrk, "A2", True)
        syrk = set_window(syrk, "C", True)

        gepp_syrk_base = self.generate_syrk_gepp_base(syrk)

        gepp_syrk_scheduled = rename(
            gepp_syrk_base, f"gepp_{self.prefix}syrk_scheduled"
        )
        gepp_syrk_scheduled = divide_loop(
            gepp_syrk_scheduled, "i", self.M_blk, ["io", "ii"], tail="cut"
        )
        gepp_syrk_scheduled = cut_loop(gepp_syrk_scheduled, "for j in _:_", 1)
        gepp_syrk_scheduled = divide_loop(
            gepp_syrk_scheduled, "j #1", self.M_blk, ["jo", "ji"], tail="cut"
        )

        gepp_syrk_scheduled = reorder_stmts(
            gepp_syrk_scheduled,
            gepp_syrk_scheduled.find("for j in _:_ #0").expand(0, 1),
        )
        gepp_syrk_scheduled = autofission(
            gepp_syrk_scheduled,
            gepp_syrk_scheduled.find("for j in _:_").after(),
            n_lifts=1,
        )
        gepp_syrk_scheduled = autofission(
            gepp_syrk_scheduled,
            gepp_syrk_scheduled.find("for j in _:_").before(),
            n_lifts=1,
        )
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)

        gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, "ii jo")
        gepp_syrk_scheduled = replace(
            gepp_syrk_scheduled, "for ii in _:_ #0", self.gebp_kernel.base_gebp
        )
        gepp_syrk_scheduled = call_eqv(
            gepp_syrk_scheduled,
            f"gebp_base_{self.gebp_kernel.this_id}(_)",
            self.gebp_kernel.scheduled_gebp,
        )
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)
        gepp_syrk_scheduled = autofission(
            gepp_syrk_scheduled,
            gepp_syrk_scheduled.find("for ii in _:_ #0").before(),
            n_lifts=1,
        )

        diag_syrk_base = rename(diag_handler, f"diag_handler")
        diag_syrk_base = diag_syrk_base.partial_eval(K=self.K_blk, N=self.M_blk)
        gepp_syrk_scheduled = replace(
            gepp_syrk_scheduled, "for ii in _:_ #1", diag_syrk_base
        )

        gebp_diag_handler = GEBP_kernel(
            self.microkernel, self.M_blk_small, self.M_blk_small, self.precision
        )
        diag_syrk_scheduled = rename(
            diag_syrk_base, f"{self.prefix}_diag_handler_scheduled"
        )
        diag_syrk_scheduled = divide_loop(
            diag_syrk_scheduled, "i", gebp_diag_handler.M_blk, ["io", "ii"], tail="cut"
        )
        diag_syrk_scheduled = divide_loop(
            diag_syrk_scheduled, "j", gebp_diag_handler.M_blk, ["jo", "ji"], tail="cut"
        )
        diag_syrk_scheduled = autofission(
            diag_syrk_scheduled,
            diag_syrk_scheduled.find("for ji in _:_ #1").before(),
            n_lifts=1,
        )
        diag_syrk_scheduled = simplify(diag_syrk_scheduled)
        diag_syrk_scheduled = reorder_loops(diag_syrk_scheduled, "ii jo")
        diag_syrk_scheduled = replace(
            diag_syrk_scheduled, "for ii in _:_ #0", gebp_diag_handler.base_gebp
        )
        diag_syrk_scheduled = call_eqv(
            diag_syrk_scheduled,
            f"gebp_base_{gebp_diag_handler.this_id}(_)",
            gebp_diag_handler.scheduled_gebp,
        )

        microkernel_diag_handler = Microkernel(
            self.machine,
            self.microkernel.M_r,
            self.microkernel.N_r,
            self.K_blk,
            self.precision,
        )
        diag_syrk_scheduled = divide_loop(
            diag_syrk_scheduled,
            "for ii in _:_",
            microkernel_diag_handler.M_r,
            ["iio", "iii"],
            tail="cut",
        )
        diag_syrk_scheduled = divide_loop(
            diag_syrk_scheduled,
            "for ji in _:_",
            microkernel_diag_handler.N_r,
            ["jio", "jii"],
            tail="cut",
        )
        diag_syrk_scheduled = autofission(
            diag_syrk_scheduled,
            diag_syrk_scheduled.find("for jii in _:_ #1").before(),
            n_lifts=1,
        )
        diag_syrk_scheduled = simplify(diag_syrk_scheduled)
        print(diag_syrk_scheduled)
        diag_syrk_scheduled = reorder_loops(diag_syrk_scheduled, "iii jio")
        diag_syrk_scheduled = replace(
            diag_syrk_scheduled,
            "for iii in _:_ #0",
            microkernel_diag_handler.base_microkernel,
        )

        diag_syrk_scheduled = call_eqv(
            diag_syrk_scheduled,
            f"microkernel_{microkernel_diag_handler.this_id}(_)",
            microkernel_diag_handler.scheduled_microkernel,
        )
        # print(simplify(diag_syrk_scheduled))

        # Unsafe microkernel
        """
        for iii in seq(0, 4):
                for jii in seq(0, (iii + 4 * iio) % 8):
                    for k in seq(0, 256):
                        C[iii + 4 * iio + 64 * io, jii + iio / 2 * 8 + 64 *
                          io] += A1[iii + 4 * iio + 64 * io,
                                    k] * A2[k, jii + iio / 2 * 8 + 64 * io]
        """

        if self.precision == "f32":  ##UNDER CONSTRUCTION
            diag_syrk_scheduled = autofission(
                diag_syrk_scheduled,
                diag_syrk_scheduled.find("for iii in _:_").before(),
                n_lifts=2,
            )
            # diag_syrk_scheduled = divide_loop(diag_syrk_scheduled, 'for jii in _:_', microkernel_diag_handler.N_r, ['jiio', 'jiii'], tail='cut')
            # diag_syrk_scheduled = autofission(diag_syrk_scheduled, diag_syrk_scheduled.find('for jiio in _:_').after(), n_lifts=2)
            diag_syrk_scheduled = simplify(diag_syrk_scheduled)
            # print(diag_syrk_scheduled)

            diag_syrk_scheduled, unsafe_microkernel_base = extract_subproc(
                diag_syrk_scheduled,
                "unsafe_microkernel_base",
                diag_syrk_scheduled.find("for io in _:_ #1"),
                order={"A1": 0, "A2": 1, "C": 2},
            )
            microkernel_diag_base = microkernel_diag_handler.base_microkernel
            microkernel_diag_scheduled = microkernel_diag_handler.scheduled_microkernel
            print(diag_syrk_scheduled)
            print(unsafe_microkernel_base)
            # print(microkernel_diag_scheduled)

            @proc
            def unsafe_microkernel_scheduled(
                A: [f32][128, 256],
                B: [f32][256, 128],
                C: [f32][128, 128],
            ):
                assert stride(C, 1) == 1
                assert stride(B, 1) == 1
                assert stride(A, 1) == 1
                # assert stride(C, 0) == 32
                # assert stride(B, 0) == 32
                # assert stride(A, 0) == 32
                # C[0, 0] = 0.0
                # A_vec: f32[4, 8] @ AVX2
                # B_vec: f32[2, 8] @ AVX2
                C_reg: f32[128, 128] @ DRAM
                for i in seq(0, 128):
                    for j in seq(0, 128):
                        C_reg[i, j] = 0.0
                for i in seq(0, 128):
                    for j in seq(0, 128):
                        for k in seq(0, 256):
                            C_reg[i, j] += A[i, k] * B[k, j]
                for i in seq(0, 128):
                    for j in seq(0, i % 16):
                        C[i, j + ((i / 16) * 16)] += C_reg[i, j + ((i / 16) * 16)]

            gebp_unsafe = GEBP_kernel(
                self.microkernel, self.M_blk, self.M_blk, self.precision
            )
            # gebp_unsafe.scheduled_gebp = inline(
            #    gebp_unsafe.scheduled_gebp,
            #    f"avx2_microkernel_4x16_{self.microkernel.this_id}(_)",
            # )

            # print(gebp_unsafe.scheduled_gebp)

            unsafe_microkernel_scheduled = replace(
                unsafe_microkernel_scheduled, "for i in _:_ #1", gebp_unsafe.base_gebp
            )
            unsafe_microkernel_scheduled = call_eqv(
                unsafe_microkernel_scheduled,
                f"gebp_base_{gebp_unsafe.this_id}(_)",
                gebp_unsafe.scheduled_gebp,
            )

            # unsafe_microkernel_scheduled = divide_loop(unsafe_microkernel_scheduled, 'j #0', self.machine.vec_width, ['jo', 'ji'], perfect=True)

            # unsafe_microkernel_scheduled = replace(
            #    unsafe_microkernel_scheduled,
            #    "for ji in _:_ #0",
            #    self.machine.set_zero_instr_f32,
            # )

            # unsafe_microkernel_scheduled = divide_loop(unsafe_microkernel_scheduled, 'j #0', self.machine.vec_width, ['jo', 'ji'], tail='cut')
            # unsafe_microkernel_scheduled = replace(
            #    unsafe_microkernel_scheduled,
            #    "for k in _:_ #1",
            #    avx2_mask_storeu_ps,
            # )
            # unsafe_microkernel_scheduled = reorder_loops(unsafe_microkernel_scheduled, "io iio")
            print(unsafe_microkernel_scheduled)

            # unsafe_microkernel_scheduled = unsafe_microkernel_scheduled.partial_eval(M=microkernel_diag_handler.M_r, N=)

            unsafe_microkernel_scheduled = set_precision(
                unsafe_microkernel_scheduled, "A", self.precision
            )
            unsafe_microkernel_scheduled = set_precision(
                unsafe_microkernel_scheduled, "B", self.precision
            )
            unsafe_microkernel_scheduled = set_precision(
                unsafe_microkernel_scheduled, "C", self.precision
            )
            unsafe_microkernel_scheduled = rename(
                unsafe_microkernel_scheduled,
                self.prefix + "_" + unsafe_microkernel_scheduled.name(),
            )

            unsafe_microkernel_base.unsafe_assert_eq(unsafe_microkernel_scheduled)

            diag_syrk_scheduled = call_eqv(
                diag_syrk_scheduled,
                "unsafe_microkernel_base",
                unsafe_microkernel_scheduled,
            )
            # diag_syrk_scheduled = inline(diag_syrk_scheduled, "s_unsafe_microkernel_scheduled(_)")

        # diag_syrk_scheduled = diag_syrk_scheduled.add_assertion("stride(A1, 0)==32")
        # diag_syrk_scheduled = diag_syrk_scheduled.add_assertion("stride(A2, 0)==32")
        # diag_syrk_scheduled = diag_syrk_scheduled.add_assertion("stride(C, 0)==32")
        # diag_syrk_scheduled.unsafe_assert_eq(diag_syrk_base)

        gepp_syrk_scheduled = call_eqv(
            gepp_syrk_scheduled, "diag_handler(_)", diag_syrk_scheduled
        )

        # gepp_syrk_scheduled = gepp_syrk_scheduled.add_assertion(
        #    f"stride(A1, 0) == {self.M_blk}"
        # )
        # gepp_syrk_scheduled = gepp_syrk_scheduled.add_assertion(f"stride(A2, 0) == N")
        # gepp_syrk_scheduled = gepp_syrk_scheduled.add_assertion(f"stride(C, 0) == N")
        # gepp_syrk_base.unsafe_assert_eq(gepp_syrk_scheduled)

        # print(gepp_syrk_scheduled)

        ### Vectorize K loop
        if self.precision == "f32" and False:
            # k_gebp = rename(self.microkernel.sgemm_window, "gebp_k_dim")
            # k_gebp = k_gebp.partial_eval(M=self.M_blk, N=1, K=self.K_blk)
            # k_gebp = reorder_loops(k_gebp, 'i j')
            k_microkernel_dim = self.e_reg

            gepp_syrk_scheduled = autofission(
                gepp_syrk_scheduled,
                gepp_syrk_scheduled.find("for ii in _:_ #0").after(),
                n_lifts=1,
            )
            gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, "ii j")
            gepp_syrk_scheduled = divide_loop(
                gepp_syrk_scheduled,
                "ii",
                k_microkernel_dim,
                ["iio", "iii"],
                perfect=True,
            )
            gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, "j iio")
            # print(gepp_syrk_scheduled)

            k_microkernel = rename(self.microkernel.sgemm_window, "k_microkernel")
            k_microkernel = k_microkernel.partial_eval(
                M=k_microkernel_dim, N=1, K=self.K_blk
            )
            k_microkernel = reorder_loops(k_microkernel, "i j")
            gepp_syrk_scheduled = replace(
                gepp_syrk_scheduled, "for j in _:_ #0", k_microkernel
            )

            k_microkernel_scheduled = rename(k_microkernel, "k_microkernel_scheduled")
            k_microkernel_scheduled = divide_loop(
                k_microkernel_scheduled,
                "i",
                self.machine.vec_width,
                ["io", "ii"],
                perfect=True,
            )
            # print(k_microkernel_scheduled)

            c_reg_str = f"C[{self.machine.vec_width}*io+ii, j]"
            k_microkernel_scheduled = stage_mem(
                k_microkernel_scheduled, "C[_] += _", c_reg_str, "C_reg"
            )
            k_microkernel_scheduled = set_memory(
                k_microkernel_scheduled, "C_reg", self.machine.mem_type
            )
            k_microkernel_scheduled = expand_dim(
                k_microkernel_scheduled,
                "C_reg",
                self.machine.vec_width,
                "ii",
                unsafe_disable_checks=True,
            )
            k_microkernel_scheduled = lift_alloc(
                k_microkernel_scheduled, "C_reg", n_lifts=4
            )
            k_microkernel_scheduled = autofission(
                k_microkernel_scheduled,
                k_microkernel_scheduled.find("C_reg[_] = _").after(),
                n_lifts=4,
            )
            k_microkernel_scheduled = autofission(
                k_microkernel_scheduled,
                k_microkernel_scheduled.find("C[_] = _").before(),
                n_lifts=4,
            )

            k_microkernel_scheduled = reorder_loops(k_microkernel_scheduled, "ii k")
            k_microkernel_scheduled = reorder_loops(k_microkernel_scheduled, "io k")
            k_microkernel_scheduled = reorder_loops(k_microkernel_scheduled, "j k")
            # print(k_microkernel_scheduled)

            # Setup A buffer in vector mem
            k_microkernel_scheduled = bind_expr(
                k_microkernel_scheduled, "A[_]", "A_vec"
            )
            k_microkernel_scheduled = set_memory(
                k_microkernel_scheduled, "A_vec", self.machine.mem_type
            )
            k_microkernel_scheduled = expand_dim(
                k_microkernel_scheduled,
                "A_vec",
                self.machine.vec_width,
                "ii",
                unsafe_disable_checks=True,
            )
            k_microkernel_scheduled = expand_dim(
                k_microkernel_scheduled,
                "A_vec",
                self.K_blk,
                "k",
                unsafe_disable_checks=True,
            )
            k_microkernel_scheduled = set_precision(
                k_microkernel_scheduled, "A_vec", self.precision
            )
            # print(k_microkernel_scheduled)

            # Setup B buffer in vector mem
            k_microkernel_scheduled = bind_expr(
                k_microkernel_scheduled, "B[_]", "B_vec"
            )

            k_microkernel_scheduled = set_memory(
                k_microkernel_scheduled, "B_vec", self.machine.mem_type
            )

            k_microkernel_scheduled = expand_dim(
                k_microkernel_scheduled,
                "B_vec",
                self.machine.vec_width,
                f"ii",
                unsafe_disable_checks=True,
            )
            k_microkernel_scheduled = expand_dim(
                k_microkernel_scheduled,
                "B_vec",
                1,
                f"j",
                unsafe_disable_checks=True,
            )
            k_microkernel_scheduled = set_precision(
                k_microkernel_scheduled, "B_vec", self.precision
            )
            # print(k_microkernel_scheduled)

            # Move A_vec and B_vec into proper sites
            k_microkernel_scheduled = lift_alloc(
                k_microkernel_scheduled, "A_vec", n_lifts=4
            )
            k_microkernel_scheduled = autofission(
                k_microkernel_scheduled,
                k_microkernel_scheduled.find("A_vec[_] = _").after(),
                n_lifts=4,
            )
            k_microkernel_scheduled = lift_alloc(
                k_microkernel_scheduled, "B_vec", n_lifts=4
            )
            k_microkernel_scheduled = autofission(
                k_microkernel_scheduled,
                k_microkernel_scheduled.find("B_vec[_] = _").after(),
                n_lifts=4,
            )
            # print(k_microkernel_scheduled)

            k_microkernel_scheduled = replace_all(
                k_microkernel_scheduled, self.machine.load_instr_f32
            )
            k_microkernel_scheduled = replace_all(
                k_microkernel_scheduled, self.machine.broadcast_instr_f32
            )
            k_microkernel_scheduled = replace_all(
                k_microkernel_scheduled, self.machine.store_instr_f32
            )
            k_microkernel_scheduled = replace_all(
                k_microkernel_scheduled, self.machine.fmadd_instr_f32
            )
            k_microkernel_scheduled = simplify(k_microkernel_scheduled)

            gepp_syrk_scheduled = call_eqv(
                gepp_syrk_scheduled, "k_microkernel(_)", k_microkernel_scheduled
            )

        return gepp_syrk_scheduled, gepp_syrk_base

    def schedule_syrk_lower_notranspose_noalpha(self, ssyrk_base: Procedure):
        syrk = divide_loop(
            ssyrk_base, "k", self.K_blk, ["ko", "ki"], tail="cut_and_guard"
        )
        syrk = autofission(syrk, syrk.find("for ko in _:_ #0").after(), n_lifts=2)
        syrk = reorder_loops(syrk, "j ko")
        syrk = reorder_loops(syrk, "i ko")
        syrk = replace(syrk, "for i in _:_ #0", self.gepp_syrk_base_lower_notranspose)
        syrk = call_eqv(
            syrk, "gepp_syrk_base(_)", self.gepp_syrk_scheduled_lower_notranspose
        )
        #        print(syrk)
        return syrk

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

        # if self.main:
        #    proc = rename(proc, proc.name() + "_main")

        return simplify(proc)

    def specialize_syrk(self, syrk: Procedure, precision: str, args=list[str]):
        prefix = "s" if precision == "f32" else "d"
        name = syrk.name().replace("exo_", "")
        syrk = rename(syrk, "exo_" + prefix + name)
        for arg in args:
            syrk = set_precision(syrk, arg, precision)
        return syrk


k_blk = 256
m_blk = 128
m_blk_small = 32
m_reg = 4
n_reg = 16
e_reg = 16


ssyrk = SYRK(C.Machine, "f32", k_blk, m_blk, m_blk_small, m_reg, n_reg, e_reg)

exo_ssyrk_lower_notranspose_noalpha_nobeta = ssyrk.entry_points[0]
exo_ssyrk_lower_notranspose_alpha_nobeta = ssyrk.entry_points[1]
exo_ssyrk_lower_notranspose_alpha_beta = ssyrk.entry_points[2]
exo_ssyrk_lower_transpose_noalpha_nobeta = ssyrk.entry_points[3]
exo_ssyrk_lower_transpose_alpha_nobeta = ssyrk.entry_points[4]
exo_ssyrk_lower_transpose_alpha_beta = ssyrk.entry_points[5]
exo_ssyrk_upper_notranspose_noalpha_nobeta = ssyrk.entry_points[6]
exo_ssyrk_upper_notranspose_alpha_nobeta = ssyrk.entry_points[7]
exo_ssyrk_upper_notranspose_alpha_beta = ssyrk.entry_points[8]
exo_ssyrk_upper_transpose_noalpha_nobeta = ssyrk.entry_points[9]
exo_ssyrk_upper_transpose_alpha_nobeta = ssyrk.entry_points[10]
exo_ssyrk_upper_transpose_alpha_beta = ssyrk.entry_points[11]
exo_ssyrk_lower_alphazero_beta = ssyrk.entry_points[12]
exo_ssyrk_upper_alphazero_beta = ssyrk.entry_points[13]

C.Machine.vec_width //= 2
dsyrk = SYRK(C.Machine, "f64", k_blk, m_blk, m_blk_small, m_reg, n_reg // 2, e_reg)
C.Machine.vec_width *= 2

exo_dsyrk_lower_notranspose_noalpha_nobeta = dsyrk.entry_points[0]
exo_dsyrk_lower_notranspose_alpha_nobeta = dsyrk.entry_points[1]
exo_dsyrk_lower_notranspose_alpha_beta = dsyrk.entry_points[2]
exo_dsyrk_lower_transpose_noalpha_nobeta = dsyrk.entry_points[3]
exo_dsyrk_lower_transpose_alpha_nobeta = dsyrk.entry_points[4]
exo_dsyrk_lower_transpose_alpha_beta = dsyrk.entry_points[5]
exo_dsyrk_upper_notranspose_noalpha_nobeta = dsyrk.entry_points[6]
exo_dsyrk_upper_notranspose_alpha_nobeta = dsyrk.entry_points[7]
exo_dsyrk_upper_notranspose_alpha_beta = dsyrk.entry_points[8]
exo_dsyrk_upper_transpose_noalpha_nobeta = dsyrk.entry_points[9]
exo_dsyrk_upper_transpose_alpha_nobeta = dsyrk.entry_points[10]
exo_dsyrk_upper_transpose_alpha_beta = dsyrk.entry_points[11]
exo_dsyrk_lower_alphazero_beta = dsyrk.entry_points[12]
exo_dsyrk_upper_alphazero_beta = dsyrk.entry_points[13]

__all__ = [p.name() for p in ssyrk.entry_points] + [
    p.name() for p in dsyrk.entry_points
]
