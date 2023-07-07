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
        self.machine = machine

        ### GEMM Kernels
        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk, precision)
        self.gebp = GEBP_kernel(self.microkernel, M_blk, N_blk, precision)
        self.gepp = GEPP_kernel(self.gebp, precision)

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

        names = [
            "gemm_alpha1_beta1_notranspose",
            "gemm_alpha1_beta1_notransa_transb",
            "gemm_alpha1_beta1_transa_notransb",
            "gemm_alpha1_beta1_transa_transb",
        ]
        scheduling_method_dict = {
            "gemm_alpha1_beta1_notranspose": self.schedule_gemm_alpha1_beta1_notranspose,
            "gemm_alpha1_beta1_notransa_transb": self.schedule_gemm_alpha1_beta1_notransa_transb,
            "gemm_alpha1_beta1_transa_notransb": self.schedule_gemm_alpha1_beta1_transa_notransb,
            "gemm_alpha1_beta1_transa_transb": self.schedule_gemm_alpha1_beta1_transa_transb,
        }

        for name in names:
            loop = gemm.find("for i in _:_ #0")
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
            tail="cut",
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

        # TODO: Encapsulate into procedure
        gemm_scheduled = inline(gemm_scheduled, call_c)
        gemm_scheduled = inline_window(gemm_scheduled, "C = C[_]")
        gemm_scheduled = inline_window(gemm_scheduled, f"A = A[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "B = B_strip[_]")
        gemm_scheduled = simplify(gemm_scheduled)

        call_c = gemm_scheduled.find(f"{self.gebp.scheduled_gebp.name()}(_)")
        gemm_scheduled = inline(gemm_scheduled, call_c)
        gemm_scheduled = inline_window(gemm_scheduled, "C = C[_]")
        gemm_scheduled = inline_window(gemm_scheduled, f"A = A[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "B = B_strip[_]")
        gemm_scheduled = simplify(gemm_scheduled)

        call_c = gemm_scheduled.find(
            f"{self.microkernel.scheduled_microkernel.name()}(_)"
        )
        gemm_scheduled = inline(gemm_scheduled, call_c)
        gemm_scheduled = inline_window(gemm_scheduled, "C = C[_]")
        gemm_scheduled = inline_window(gemm_scheduled, f"A = A[_]")
        gemm_scheduled = inline_window(gemm_scheduled, "B = B_reg_strip[_]")
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

        # Edge case handling
        # Handle as much as possible with normal microkernel
        gemm_scheduled = divide_loop(
            gemm_scheduled, "ii", self.microkernel.M_r, ["iio", "iii"], tail="cut"
        )
        gemm_scheduled = divide_loop(
            gemm_scheduled, "j", self.microkernel.N_r, ["jo", "ji"], tail="cut"
        )
        gemm_scheduled = simplify(gemm_scheduled)
        gemm_scheduled = reorder_loops(gemm_scheduled, "iii jo")
        gemm_scheduled = replace(
            gemm_scheduled, "for iii in _:_ #0", self.microkernel.base_microkernel
        )
        call_c = gemm_scheduled.find(f"microkernel_{self.microkernel.this_id}(_)")
        gemm_scheduled = call_eqv(
            gemm_scheduled, call_c, self.microkernel.scheduled_microkernel
        )

        # Now, handle the remaining portions
        gemm_scheduled = divide_loop(
            gemm_scheduled, "j", self.microkernel.N_r, ["jo", "ji"], tail="cut"
        )
        gemm_scheduled = simplify(gemm_scheduled)
        gemm_scheduled = reorder_loops(gemm_scheduled, "iii jo")
        edge_microkernels = {}
        for i in range(1, self.microkernel.M_r):
            edge_microkernels[i] = Microkernel(
                self.machine,
                i,
                self.microkernel.N_r,
                self.microkernel.K_blk,
                self.precision,
            )
        gemm_scheduled = specialize(
            gemm_scheduled,
            "for iii in _:_",
            [
                f"M % {self.gebp.M_blk} % {self.microkernel.M_r} == {i}"
                for i in range(1, self.microkernel.M_r)
            ],
        )
        gemm_scheduled = simplify(gemm_scheduled)
        for i in range(1, self.microkernel.M_r):
            loop_c = gemm_scheduled.find(f"for iii in _:_")
            gemm_scheduled = replace(
                gemm_scheduled, loop_c, edge_microkernels[i].base_microkernel
            )
            call_c = gemm_scheduled.find(
                f"microkernel_{edge_microkernels[i].this_id}(_)"
            )
            gemm_scheduled = call_eqv(
                gemm_scheduled, call_c, edge_microkernels[i].scheduled_microkernel
            )

        # Do zero padding to handle other edge case
        # Handle as much as possible with microkernel
        # TODO: This is very redundant w/ other code blocks above. Should make this a composed schedule
        gemm_scheduled = divide_loop(
            gemm_scheduled, "i #0", self.microkernel.M_r, ["io", "ii"], tail="cut"
        )
        gemm_scheduled = divide_loop(
            gemm_scheduled, "ji #1", self.microkernel.N_r, ["jio", "jii"], tail="cut"
        )
        loop_c = gemm_scheduled.find("for jii in _:_ #1")
        gemm_scheduled = autofission(gemm_scheduled, loop_c.before(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, "ii jio")
        gemm_scheduled = replace(
            gemm_scheduled, "for ii in _:_ #0", self.microkernel.base_microkernel
        )
        call_c = gemm_scheduled.find(f"microkernel_{self.microkernel.this_id}(_)")
        gemm_scheduled = call_eqv(
            gemm_scheduled, call_c, self.microkernel.scheduled_microkernel
        )

        sched_zpad, base_zpad = self.microkernel.generate_microkernel_zpad(
            self.machine,
            self.microkernel.M_r,
            self.microkernel.N_r,
            self.microkernel.K_blk,
            self.gebp.M_blk,
        )
        loop_c = gemm_scheduled.find("for ii in _:_ #0")
        gemm_scheduled = replace(gemm_scheduled, loop_c, base_zpad)
        call_c = gemm_scheduled.find(
            f"microkernelzpad_{self.microkernel.microkernel_id}(_)"
        )
        gemm_scheduled = call_eqv(gemm_scheduled, call_c, sched_zpad)

        return gemm_scheduled
        # TODO: Schedule each of the tranpose variants

    def schedule_gemm_alpha1_beta1_notransa_transb(self, gemm_procedure: Procedure):
        # TODO
        return gemm_procedure

    def schedule_gemm_alpha1_beta1_transa_notransb(self, gemm_procedure: Procedure):
        # TODO
        return gemm_procedure

    def schedule_gemm_alpha1_beta1_transa_transb(self, gemm_procedure: Procedure):
        # TODO
        return gemm_procedure

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
        name = gemm.name().replace("_base_", "_")
        spec = rename(gemm, "exo_" + self.prefix + name)

        for arg in args:
            spec = set_precision(spec, arg, self.precision)

        return spec


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
n_blk = C.gemm.n_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg

sgemm = GEMM(C.Machine, "f32", k_blk, m_blk, n_blk, m_reg, n_reg)

exo_sgemm_alpha1_beta1 = sgemm.entry_points[0]
exo_sgemm_alphaneg1_beta1 = sgemm.entry_points[1]
exo_sgemm_alpha1_betaneg1 = sgemm.entry_points[2]
exo_sgemm_alphaneg1_betaneg1 = sgemm.entry_points[3]

__all__ = [p.name() for p in sgemm.entry_points]
