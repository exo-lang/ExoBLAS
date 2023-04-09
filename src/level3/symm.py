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

from kernels.gemm_kernels import GEBP_kernel, GEPP_kernel, Microkernel
from format_options import *

import exo_blas_config as C

class SYMM:

    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 K_blk: int, M_blk: int, 
                 M_r: int, N_r: int, main=True):

                self.K_blk = K_blk
                self.M_blk = M_blk
                self.M_r = M_r
                self.N_r = N_r
                self.machine = machine
                self.precision = precision
                self.main = main

                self.microkernel = Microkernel(machine, M_r, N_r, K_blk, precision)
                self.gebp = GEBP_kernel(self.microkernel, M_blk, precision)
                self.gepp = GEPP_kernel(self.gebp, precision)

                ### Base Procedures

                @proc
                def symm_lower_left_noalpha_nobeta(
                     M: size,
                     N: size,
                     K: size,
                     A: f32[M, K],
                     B: f32[K, N],
                     C: f32[M, N]
                 ):
                    #This is a brute force method that just does GEMM. Let's see if this is okay performance wise
                    for i in seq(0, M):
                        for j in seq(0, N):
                            for k in seq(0, K):
                                C[i, j] += A[i, k] * B[k, j]

                symm_lower_left_noalpha_nobeta = self.specialize_symm(symm_lower_left_noalpha_nobeta, self.precision, ['A', 'B', 'C'])
                scheduled_symm = self.schedule_symm_lower_noalpha(symm_lower_left_noalpha_nobeta)

                self.entry_points = [scheduled_symm]


    def schedule_symm_lower_noalpha(self, symm):

        symm = divide_loop(symm, 'for k in _:_', self.K_blk, ['ko', 'ki'], tail='cut')
        symm = autofission(symm, symm.find('for ko in _:_ #0').after(), n_lifts=2)
        symm = reorder_loops(symm, 'j ko')
        symm = reorder_loops(symm, 'i ko')
        symm = replace_all(symm, self.gepp.gepp_base)
        symm = call_eqv(symm, f'gepp_base_{self.gepp.this_id}(_)', self.gepp.gepp_scheduled)

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



k_blk = C.symm.k_blk
m_blk = C.symm.m_blk
m_reg = C.symm.m_reg
n_reg = C.symm.n_reg


ssymm = SYMM(C.Machine, 'f32',
             k_blk, m_blk, 
             m_reg, n_reg
             )

exo_ssymm_lower_left_noalpha_nobeta_main = ssymm.entry_points[0]

__all__ = [p.name() for p in ssymm.entry_points]