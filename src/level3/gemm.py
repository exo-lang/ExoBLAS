from __future__ import annotations
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
    ):
        self.K_blk = K_blk
        self.M_blk = M_blk
        self.N_blk = N_blk
        self.M_reg = M_reg
        self.N_reg = N_reg

        self.precision = precision
        self.prefix = "s" if self.precision == "f32" else "d"

        ### Base GEMM procedures

        @proc
        def gemm_base_alpha1_beta1(
            M: size,
            N: size,
            K: size,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
            transa: size,
            transb: size,
        ):
            assert M == N == K

            if transa == 0:
                if transb == 0:
                    # C += A*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[k, j]
                else:
                    # C += A*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[j, k]
            else:
                if transb == 0:
                    # C += A^T*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[k, j]
                else:
                    # C += A^T*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[j, k]

        @proc
        def gemm_base_alphaneg1_beta1(
            M: size,
            N: size,
            K: size,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
            transa: size,
            transb: size,
        ):
            assert M == N == K

            alpha: f32[1]
            alpha[0] = -1.0

            if transa == 0:
                if transb == 0:
                    # C += A*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[k, j] * alpha[0]
                else:
                    # C += A*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[j, k] * alpha[0]
            else:
                if transb == 0:
                    # C += A^T*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[k, j] * alpha[0]
                else:
                    # C += A^T*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[j, k] * alpha[0]

        @proc
        def gemm_base_alpha1_betaneg1(
            M: size,
            N: size,
            K: size,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
            transa: size,
            transb: size,
        ):
            assert M == N == K

            beta: f32[1]
            beta[0] = -1.0

            if transa == 0:
                if transb == 0:
                    # C += A*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[k, j]
                            C[i, j] = C[i, j] * beta[0]
                else:
                    # C += A*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[j, k]
                            C[i, j] = C[i, j] * beta[0]
            else:
                if transb == 0:
                    # C += A^T*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[k, j]
                            C[i, j] = C[i, j] * beta[0]
                else:
                    # C += A^T*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[j, k]
                            C[i, j] = C[i, j] * beta[0]

        @proc
        def gemm_base_alphaneg1_betaneg1(
            M: size,
            N: size,
            K: size,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM,
            transa: size,
            transb: size,
        ):
            assert M == N == K

            alpha: f32[1]
            alpha[0] = -1.0
            beta: f32[1]
            beta[0] = -1.0

            if transa == 0:
                if transb == 0:
                    # C += A*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[k, j] * alpha[0]
                            C[i, j] = C[i, j] * beta[0]
                elif transb == 1:
                    # C += A*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[j, k] * alpha[0]
                            C[i, j] = C[i, j] * beta[0]
            elif transa == 1:
                if transb == 0:
                    # C += A^T*B
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[k, j] * alpha[0]
                            C[i, j] = C[i, j] * beta[0]
                elif transb == 1:
                    # C += A^T*B^T
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[k, i] * B[j, k] * alpha[0]
                            C[i, j] = C[i, j] * beta[0]

        # Schedule each base gemm procedure
        exo_gemm_alpha1_beta1 = self.specialize_gemm(
            gemm_base_alpha1_beta1, ["A", "B", "C"]
        )
        exo_gemm_alpha1_beta1 = self.schedule_gemm_alpha1_beta1(exo_gemm_alpha1_beta1)

        exo_gemm_alphaneg1_beta1 = self.specialize_gemm(
            gemm_base_alphaneg1_beta1, ["A", "B", "C", "alpha"]
        )
        # TODO: Schedule this variant

        exo_gemm_alpha1_betaneg1 = self.specialize_gemm(
            gemm_base_alpha1_betaneg1, ["A", "B", "C", "beta"]
        )
        # TODO: Schedule this variant

        exo_gemm_alphaneg1_betaneg1 = self.specialize_gemm(
            gemm_base_alphaneg1_betaneg1, ["A", "B", "C", "alpha", "beta"]
        )
        # TODO: Schedule this variant

        self.entry_points = [
            exo_gemm_alpha1_beta1,
            exo_gemm_alphaneg1_beta1,
            exo_gemm_alpha1_betaneg1,
            exo_gemm_alphaneg1_betaneg1,
        ]

    def schedule_gemm_alpha1_beta1(self, gemm: Procedure):

        notranspose_loop = gemm.find("for i in _:_ #0")
        notransa_transb_loop = gemm.find("for i in _:_ #1")
        transa_notransb_loop = gemm.find("for i in _:_ #2")
        transa_transb_loop = gemm.find("for i in _:_ #3")

        loops = [
            notranspose_loop,
            notransa_transb_loop,
            transa_notransb_loop,
            transa_transb_loop,
        ]
        names = [
            "gemm_alpha1_beta1_notranspose",
            "gemm_alpha1_beta1_notransa_transb",
            "gemm_alpha1_beta1_transa_notransb",
            "gemm_alpha1_beta1_transa_transb",
        ]
        scheduling_method_dict = {
            "gemm_alpha1_beta1_notranspose": self.schedule_gemm_alpha1_beta1_notranspose
            # TODO: Add other scheduling methods
        }

        for loop, name in zip(loops, names):
            gemm, variant_base = extract_subproc(gemm, name, loop)
            scheduled_variant = scheduling_method_dict[name](variant_base)
            gemm = call_eqv(gemm, f"{name}(_)", scheduled_variant)

        return gemm

    def schedule_gemm_alpha1_beta1_notranspose(self, gemm_procedure: Procedure):

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
        # TODO: Schedule each of the tranpose variants

    def schedule_gemm_alphaneg1_beta1(self, gemm: Procedure):

        # Schedule each of the tranpose variants
        return gemm

    def schedule_gemm_alpha1_betaneg1(self, gemm: Procedure):

        # Schedule each of the tranpose variants
        return gemm

    def schedule_gemm_alphaneg1_betaneg1(self, gemm: Procedure):

        # Schedule each of the tranpose variants
        return gemm

    def specialize_gemm(self, gemm: Procedure, args: list[str]):
        name = gemm.name().replace("_base_", "")
        spec = rename(gemm, "exo_" + self.prefix + name)

        for arg in args:
            spec = set_precision(spec, arg, self.precision)

        return spec


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
n_blk = C.gemm.n_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg

gemm = GEMM(C.Machine, "f32", k_blk, m_blk, n_blk, m_reg, n_reg)

exo_gemm_alpha1_beta1 = gemm.entry_points[0]
exo_gemm_alphaneg1_beta1 = gemm.entry_points[1]
exo_gemm_alpha1_betaneg1 = gemm.entry_points[2]
exo_gemm_alphaneg1_betaneg1 = gemm.entry_points[3]

__all__ = [p.name() for p in gemm.entry_points]
