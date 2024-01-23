from __future__ import annotations
from os import abort

import sys
import getopt
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *
from exo.libs.memories import DRAM_STATIC

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEBP_kernel, GEPP_kernel, Microkernel
from format_options import *

import exo_blas_config as C
from composed_schedules import *


class SYMM:
    def __init__(
        self,
        machine: "MachineParameters",
        precision: str,
        K_blk: int,
        M_blk: int,
        N_blk: int,
        M_r: int,
        N_r: int,
        do_rename=False,
        main=True,
    ):

        self.K_blk = K_blk
        self.M_blk = M_blk
        self.N_blk = N_blk
        self.M_r = M_r
        self.N_r = N_r
        self.machine = machine
        self.precision = precision
        self.main = main

        self.microkernel = Microkernel(machine, M_r, N_r, K_blk, precision)
        self.gebp = GEBP_kernel(self.microkernel, M_blk, N_blk, precision)
        self.gepp = GEPP_kernel(self.gebp, precision)

        ### Base Procedures

        @proc
        def symm_lower_left_noalpha_nobeta(
            M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]
        ):
            # This is a brute force method that just does GEMM. Let's see if this is okay performance wise
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * B[k, j]

        symm_lower_left_noalpha_nobeta = self.specialize_symm(
            symm_lower_left_noalpha_nobeta, self.precision, ["A", "B", "C"]
        )
        scheduled_symm = self.schedule_symm_lower_noalpha(
            symm_lower_left_noalpha_nobeta
        )

        self.entry_points = [scheduled_symm]

        if do_rename:
            for i in range(len(self.entry_points)):
                self.entry_points[i] = rename(
                    self.entry_points[i],
                    f"{self.entry_points[i].name()}_{N_blk}_{M_blk}_{K_blk}",
                )

    def schedule_symm_lower_noalpha(self, symm):

        symm = divide_loop(symm, "for k in _:_", self.K_blk, ["ko", "ki"], tail="cut")
        symm = autofission(symm, symm.find("for ko in _:_ #0").after(), n_lifts=2)
        symm = reorder_loops(symm, "j ko")
        symm = reorder_loops(symm, "i ko")

        symm = divide_loop(
            symm, "for j in _:_ #0", self.N_blk, ["jo", "ji"], tail="cut"
        )
        symm = autofission(symm, symm.find("for jo in _:_ #0").after(), n_lifts=2)
        symm = reorder_loops(symm, "i jo")
        symm = reorder_loops(symm, "ko jo")

        symm = stage_mem(
            symm,
            "for i in _:_ #0",
            f"B[{self.microkernel.K_blk}*ko:{self.microkernel.K_blk}*ko+{self.microkernel.K_blk}, {self.gebp.N_blk}*jo:{self.gebp.N_blk}*jo+{self.gebp.N_blk}]",
            "B_strip",
        )

        symm = replace_all_stmts(symm, self.gepp.gepp_base)
        call_c = symm.find(f"gepp_base_{self.gepp.this_id}(_)")
        symm = call_eqv(
            symm, f"gepp_base_{self.gepp.this_id}(_)", self.gepp.gepp_scheduled
        )

        symm = call_eqv(
            symm,
            call_c,
            self.gepp.gepp_scheduled,
        )
        symm = inline(symm, call_c)
        symm = inline_window(symm, "C = C[_]")
        symm = inline_window(symm, f"A = A[_]")
        symm = inline_window(symm, "B = B_strip[_]")
        symm = simplify(symm)

        while True:
            try:
                symm = lift_alloc(symm, "B_reg_strip:_")
            except:
                break
        while True:
            try:
                symm = lift_alloc(symm, "B_strip:_")
            except:
                break
        symm = set_memory(symm, "B_strip:_", DRAM_STATIC)
        symm = set_memory(symm, "B_reg_strip:_", DRAM_STATIC)

        return simplify(symm)

    def specialize_symm(self, symm, precision, args):
        prefix = "s" if precision == "f32" else "d"
        name = symm.name().replace("exo_", "")
        specialized = rename(symm, "exo_" + prefix + name)

        for arg in args:
            specialized = set_precision(specialized, arg, precision)

        if self.main:
            specialized = rename(specialized, specialized.name() + "_main")

        return specialized


k_blk = [48, 48 * 2, 48 * 4, 48 * 8, 480, 480]
m_blk = [48, 48 * 2, 48 * 4, 48 * 8, 240, 240]
n_blk = [48, 48 * 2, 48 * 4, 48 * 8, 480, 960]
m_reg = 6
n_reg = 16


ssymm_kernels = [
    SYMM(C.Machine, "f32", k, m, n, m_reg, n_reg, True, False)
    for (k, m, n) in zip(k_blk, m_blk, n_blk)
]

exo_ssymm_lower_left_noalpha_nobeta_48_48_48 = ssymm_kernels[0].entry_points[0]
exo_ssymm_lower_left_noalpha_nobeta_96_96_96 = ssymm_kernels[1].entry_points[0]
exo_ssymm_lower_left_noalpha_nobeta_192_192_192 = ssymm_kernels[2].entry_points[0]
exo_ssymm_lower_left_noalpha_nobeta_384_384_384 = ssymm_kernels[3].entry_points[0]
exo_ssymm_lower_left_noalpha_nobeta_480_240_480 = ssymm_kernels[4].entry_points[0]
exo_ssymm_lower_left_noalpha_nobeta_960_240_480 = ssymm_kernels[5].entry_points[0]

ssymm_kernel_names = []
for s in ssymm_kernels:
    ssymm_kernel_names.extend(s.entry_points)

__all__ = [p.name() for p in ssymm_kernel_names]
