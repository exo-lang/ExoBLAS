from __future__ import annotations

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEBP_kernel, Microkernel, NeonMachine, MachineParameters
from format_options import *

import exo_blas_config as C

class SYRK:
    """
    TODO: Add Beta and Alpha
    """
    def __init__(self, machine: MachineParameters,
                 uplo: ExoBlasUplo,
                 transpose: ExoBlasT,  
                 K_blk: int, M_blk: int, 
                 M_r: int, N_r: int):
        
        # Generate kernels
        self.microkernel = Microkernel(machine, M_r, N_r, K_blk)
        self.gebp_kernel = GEBP_kernel(self.microkernel, M_blk)

        # Blocking dimensions
        self.K_blk = K_blk
        self.M_blk = M_blk

        @proc
        def syrk_lower_notranspose(N: size, 
                 K: size, 
                 A1: f32[N, K] @ DRAM, 
                 A2: f32[K, N] @ DRAM,
                 C: f32[N, N] @ DRAM):
            # C = A*A**T + C      
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1

            for i in seq(0, N):
                for j in seq(0, i+1):
                    for k in seq(0, K):
                        C[i, j] += A1[i, k]*A2[k, j]

        @proc
        def syrk_lower_transpose(N: size, 
                           K: size, 
                           A1: f32[K, N] @ DRAM, 
                           A2: f32[N, K] @ DRAM,
                           C: f32[N, N] @ DRAM):
            # C = A**T*A + C      
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1    

            for j in seq(0, N):
                for i in seq(0, j+1):
                    for k in seq(0, K):
                        C[i, j] += A2[j, k]*A1[k, i]

        if uplo==ExoBlasLower:
            
            if transpose==ExoBlasNoTranspose:
                self.syrk_base = syrk_lower_notranspose

                self.syrk_win = rename(self.syrk_base, "syrk_win")
                self.syrk_win = set_window(self.syrk_win, 'A1', True)
                self.syrk_win = set_window(self.syrk_win, 'A2', True)
                self.syrk_win = set_window(self.syrk_win, 'C', True)

                self.gepp_syrk_scheduled, self.gepp_syrk_base = self.generate_syrk_gepp_lower_notranspose()
                self.syrk_scheduled = self.schedule_syrk_gepp_lower_notranspose()

            if transpose==ExoBlasTranspose:
                self.syrk_base = syrk_lower_notranspose

                self.syrk_win = rename(self.syrk_base, "syrk_win")
                self.syrk_win = set_window(self.syrk_win, 'A1', True)
                self.syrk_win = set_window(self.syrk_win, 'A2', True)
                self.syrk_win = set_window(self.syrk_win, 'C', True)
                #TODO:
                raise Exception("NOT SUPPORTED")

    
    def generate_syrk_gepp_base(self):
        gepp_syrk_base = rename(self.syrk_win, "gepp_syrk_base")
        gepp_syrk_base = gepp_syrk_base.partial_eval(K=self.microkernel.K_blk)
        return gepp_syrk_base
    

    def generate_syrk_gepp_lower_notranspose(self):

        gepp_syrk_base = self.generate_syrk_gepp_base()

        gepp_syrk_scheduled = rename(gepp_syrk_base, "gepp_syrk_scheduled")
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'i', self.M_blk, ['io', 'ii'], tail='cut_and_guard')
        gepp_syrk_scheduled = cut_loop(gepp_syrk_scheduled, 'for j in _:_', 1)
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'j #1', self.M_blk, ['jo', 'ji'], tail='cut_and_guard')

        gepp_syrk_scheduled = reorder_stmts(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_ #0').expand(1))
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').after(), n_lifts=1)
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').before(), n_lifts=1)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)

        gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, 'ii jo')
        gepp_syrk_scheduled = replace(gepp_syrk_scheduled, 'for ii in _:_ #0', self.gebp_kernel.base_gebp)
        gepp_syrk_scheduled = call_eqv(gepp_syrk_scheduled, f'gebp_base_{self.gebp_kernel.this_id}(_)', self.gebp_kernel.scheduled_gebp)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)

        return gepp_syrk_scheduled, gepp_syrk_base

    
    def schedule_syrk_gepp_lower_notranspose(self):
        syrk = divide_loop(self.syrk_base, 'k', self.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        syrk = autofission(syrk, syrk.find('for ko in _:_ #0').after(), n_lifts=2)
        syrk = reorder_loops(syrk, 'j ko')
        syrk = reorder_loops(syrk, 'i ko')
        syrk = replace(syrk, 'for i in _:_ #0', self.gepp_syrk_base)
        syrk = call_eqv(syrk, 'gepp_syrk_base(_)', self.gepp_syrk_scheduled)
        return syrk


k_blk = C.syrk.k_blk
m_blk = C.syrk.m_blk
m_reg = C.syrk.m_reg
n_reg = C.syrk.n_reg

ssyrk = SYRK(NeonMachine, 
             ExoBlasLower, ExoBlasNoTranspose, 
             k_blk, m_blk, 
             m_reg, n_reg
             )

scheduled = ssyrk.syrk_scheduled

__all__ = ["scheduled"]
