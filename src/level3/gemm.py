from __future__ import annotations

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from kernels.format_options import *

import exo_blas_config as C

class GEMM:

    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 transa: ExoBlasT, transb: ExoBlasT,
                 alpha: float, beta: float,
                 K_blk: int, M_blk: int, 
                 M_reg: int, N_reg: int):

        ### GEMM PROCEDURES
        @proc
        def gemm_notranspose(
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

        ### ALPHA AND BETA
        #@proc
        #def sgemm_                        

        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk) #TODO: Add precision args to microkernel, gebp, gepp
        self.gebp = GEBP_kernel(self.microkernel, M_blk)
        self.gepp = GEPP_kernel(self.gebp)
        self.gemm_base = gemm_notranspose
        self.gemm_scheduled = self.schedule_gepp_notranspose()


    def schedule_gepp_notranspose(self):
        gemm_scheduled = divide_loop(self.gemm_base, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')
        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)
        return gemm_scheduled



k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg

if C.gemm.transa:
    transa = ExoBlasTranspose
else:
    transa = ExoBlasNoTranspose

if C.gemm.transb:
    transb = ExoBlasTranspose
else:
    transb = ExoBlasNoTranspose

alpha = C.gemm.alpha
beta = C.gemm.beta

gemm = GEMM(
    C.Machine,
    'f32', 
    transa, transb, 
    alpha, beta, 
    k_blk, m_blk, 
    m_reg, n_reg
)

gemm = gemm.gemm_scheduled

__all__ = ['gemm']
