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

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from syrk import SYRK
from format_options import *

import exo_blas_config as C



class SYR2K:

    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 K_blk: int, M_blk: int,
                 M_r: int, N_r: int, main:bool=True):
        
        self.main = main

        self.syrk = SYRK(machine, precision, K_blk, M_blk, M_r, N_r)
        syrk_scheduled = self.syrk.entry_points[0]
        

        ### Matrix addition
        @proc
        def mat_add(M:size, N:size, A: f32[M, N], B: f32[M, N], C: f32[M, N]):
            for i in seq(0, M):
                for j in seq(0, N):
                    C[i, j] += A[i, j] + B[i, j]
        
        ### SYR2K BASE
        @proc
        def syr2k_lower_notranspose_noalpha(
            M: size,
            N: size,
            A: f32[M, N],
            alpha: f32[1],
            B: f32[N, M],
            beta: f32[1],
            C: f32[M, M],
        ):  
            assert M==N
        
            syrk_scheduled(M, N, alpha, A, B, beta, C)
            syrk_scheduled(N, M, alpha, A, B, beta, C)
        syr2k_lower_notranspose_noalpha = self.specialize_sy2rk(syr2k_lower_notranspose_noalpha, precision, ["A", "B", "C", "alpha", "beta"])
        print(self.syrk.microkernel.scheduled_microkernel)
        
        self.entry_points = [syr2k_lower_notranspose_noalpha]
    
    def specialize_sy2rk(self, syr2k: Procedure, precision: str, args=list[str]):
        prefix = "s" if precision == "f32" else "d"
        name = syr2k.name().replace("exo_", "")
        syr2k = rename(syr2k, "exo_" + prefix + name)
        if self.main:
            syr2k = rename(syr2k, syr2k.name() + "_main")
        for arg in args:
            syr2k = set_precision(syr2k, arg, precision)
        return syr2k

k_blk = C.syrk.k_blk
m_blk = C.syrk.m_blk
m_reg = C.syrk.m_reg
n_reg = C.syrk.n_reg

ssyr2k = SYR2K(C.Machine, 'f32',
                k_blk, m_blk,
                m_reg, n_reg)

exo_ssyr2k_lower_notranspose_noalpha_main = ssyr2k.entry_points[0]

__all__ = [p.name() for p in ssyr2k.entry_points]

