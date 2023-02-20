from __future__ import annotations

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from format_options import *

import exo_blas_config as C

class GEMM:

    def __init__(self, machine: "MachineParameters",
                 transa: ExoBlasT, transb: ExoBlasT,
                 alpha: int, beta: int,
                 K_blk: int, M_blk: int, 
                 M_reg: int, N_reg: int):

        @proc
        def sgemm_notranspose(
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
                        C[i, j] += A[i,k] * B[k,j] 

        @proc
        def sgemm_b_transpose(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[N, K] @ DRAM,
        ):
            assert(N==K)
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i,k] * B[j,k]                         

        self.microkernel = Microkernel(machine, transa, transb, M_reg, N_reg, K_blk)
        self.gebp = GEBP_kernel(self.microkernel, transa, transb, M_blk)
        self.gepp = GEPP_kernel(self.gebp, transa, transb)
        self.sgemm_base = sgemm_notranspose
        self.sgemm_scheduled = self.schedule_gepp_notranspose()


    def schedule_gepp_notranspose(self):
        sgemm_scheduled = divide_loop(self.sgemm_base, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        sgemm_scheduled = autofission(sgemm_scheduled, sgemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        sgemm_scheduled = reorder_loops(sgemm_scheduled, 'j ko')
        sgemm_scheduled = reorder_loops(sgemm_scheduled, 'i ko')
        sgemm_scheduled = replace(sgemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        sgemm_scheduled = call_eqv(sgemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)
        return sgemm_scheduled


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg
#TODO: Add transpose options

sgemm = GEMM(
    C.Machine, 
    ExoBlasNoTranspose, ExoBlasNoTranspose, 
    1, 1, 
    k_blk, m_blk, 
    m_reg, n_reg
)

sgemm = sgemm.sgemm_scheduled

__all__ = ['sgemm']
