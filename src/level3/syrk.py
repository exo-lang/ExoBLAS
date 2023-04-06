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

import exo_blas_config as C

class SYRK:
    """
    TODO: Add Beta and Alpha
    """
    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 K_blk: int, M_blk: int, 
                 M_r: int, N_r: int):
        
        # Precision
        self.precision = precision
        self.prefix = 's' if precision=='f32' else 'd'

        # Generate kernels
        self.microkernel = Microkernel(machine, M_r, N_r, K_blk, self.precision)
        self.gebp_kernel = GEBP_kernel(self.microkernel, M_blk, self.precision)

        # Blocking dimensions
        self.K_blk = K_blk
        self.M_blk = M_blk

        #machine
        self.machine = machine

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
        syrk_lower_notranspose = set_precision(syrk_lower_notranspose, "A1", self.precision)
        syrk_lower_notranspose = set_precision(syrk_lower_notranspose, "A2", self.precision)
        syrk_lower_notranspose = set_precision(syrk_lower_notranspose, "C", self.precision)
        syrk_lower_notranspose = rename(syrk_lower_notranspose, f"{self.prefix}{syrk_lower_notranspose.name()}")
        
        @proc
        def diag_handler_lower_notranspose(N: size, 
                 K: size, 
                 A1: [f32][N, K] @ DRAM, 
                 A2: [f32][K, N] @ DRAM,
                 C: [f32][N, N] @ DRAM):
            # C = A*A**T + C      
            assert N >= 1
            assert K >= 1
            assert stride(A1, 1) == 1
            assert stride(A2, 1) == 1
            assert stride(C, 1) == 1

            for i in seq(0, N):
                for j in seq(0, i):
                    for k in seq(0, K):
                        C[i, j] += A1[i, k]*A2[k, j]
        diag_handler_lower_notranspose = set_precision(diag_handler_lower_notranspose, "A1", self.precision)
        diag_handler_lower_notranspose = set_precision(diag_handler_lower_notranspose, "A2", self.precision)
        diag_handler_lower_notranspose = set_precision(diag_handler_lower_notranspose, "C", self.precision)
        diag_handler_lower_notranspose = rename(diag_handler_lower_notranspose, f"{self.prefix}_{diag_handler_lower_notranspose.name()}")

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


        self.syrk_win_lower_notranspose = rename(syrk_lower_notranspose, "syrk_win")
        self.syrk_win_lower_notranspose = set_window(self.syrk_win_lower_notranspose, 'A1', True)
        self.syrk_win_lower_notranspose = set_window(self.syrk_win_lower_notranspose, 'A2', True)
        self.syrk_win_lower_notranspose = set_window(self.syrk_win_lower_notranspose, 'C', True)
        self.gepp_syrk_scheduled_lower_notranspose, self.gepp_syrk_base_lower_notranspose = self.generate_syrk_gepp_lower_notranspose(diag_handler_lower_notranspose)
        syrk_scheduled_lower_notranspose = self.schedule_syrk_lower_notranspose(syrk_lower_notranspose)

        @proc
        def exo_syrk_lower_notranspose(
            N: size,
            K: size,
            alpha: f32[1] @ DRAM,
            A1: f32[N, K] @ DRAM,
            A2: f32[K, N] @ DRAM,
            beta: f32[1] @ DRAM,
            C: f32[N, N] @ DRAM
        ):
            syrk_scheduled_lower_notranspose(N, K, A1, A2, C)
        exo_syrk_lower_notranspose = self.specialize_syrk(exo_syrk_lower_notranspose, self.precision)

        self.entry_points = [
            exo_syrk_lower_notranspose
        ]

    
    def generate_syrk_gepp_base(self, syrk_win: Procedure):
        gepp_syrk_base = rename(syrk_win, "gepp_syrk_base")
        gepp_syrk_base = gepp_syrk_base.partial_eval(K=self.microkernel.K_blk)
        return gepp_syrk_base
    

    def generate_syrk_gepp_lower_notranspose(self, diag_handler: Procedure):

        #assert(self.M_blk >= 128) # Temporary

        gepp_syrk_base = self.generate_syrk_gepp_base(self.syrk_win_lower_notranspose)

        gepp_syrk_scheduled = rename(gepp_syrk_base, f"gepp_{self.prefix}syrk_scheduled")
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'i', self.M_blk, ['io', 'ii'], tail='cut')
        gepp_syrk_scheduled = cut_loop(gepp_syrk_scheduled, 'for j in _:_', 1)
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'j #1', self.M_blk, ['jo', 'ji'], tail='cut')

        gepp_syrk_scheduled = reorder_stmts(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_ #0').expand(1))
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').after(), n_lifts=1)
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').before(), n_lifts=1)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)

        gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, 'ii jo')
        gepp_syrk_scheduled = replace(gepp_syrk_scheduled, 'for ii in _:_ #0', self.gebp_kernel.base_gebp)
        gepp_syrk_scheduled = call_eqv(gepp_syrk_scheduled, f'gebp_base_{self.gebp_kernel.this_id}(_)', self.gebp_kernel.scheduled_gebp)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for ii in _:_ #0').before(), n_lifts=1)


        #gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'ii #1', self.microkernel.M_r, ['iii', 'iio'], perfect=True)
        #gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'ji', self.microkernel.N_r, ['jii', 'jio'], tail='cut')
        #gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)
        #gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for jii in _:_').after(), n_lifts=1)
        #gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, 'iio jii')
        #gepp_syrk_scheduled = replace(gepp_syrk_scheduled, 'for iio in _:_ #0', self.microkernel.base_microkernel)
        #gepp_syrk_scheduled = call_eqv(gepp_syrk_scheduled, f'microkernel_{self.microkernel.this_id}(_)', self.microkernel.scheduled_microkernel)
        #print(gepp_syrk_scheduled)
        

        #return gepp_syrk_scheduled, gepp_syrk_base

        diag_syrk_base = rename(diag_handler, "diag_handler")
        diag_syrk_base = diag_syrk_base.partial_eval(K=self.K_blk, N=self.M_blk)
        gepp_syrk_scheduled = replace(gepp_syrk_scheduled, 'for ii in _:_ #1', diag_syrk_base)
        
        gebp_diag_handler = GEBP_kernel(self.microkernel, self.M_blk//2, self.precision)
        diag_syrk_scheduled = rename(diag_syrk_base, f'{self.prefix}_diag_handler_scheduled')
        diag_syrk_scheduled = divide_loop(diag_syrk_scheduled, 'i', gebp_diag_handler.M_blk, ['io', 'ii'], tail='cut')
        diag_syrk_scheduled = divide_loop(diag_syrk_scheduled, 'j', gebp_diag_handler.M_blk, ['jo', 'ji'], tail='cut')
        diag_syrk_scheduled = autofission(diag_syrk_scheduled, diag_syrk_scheduled.find('for ji in _:_ #1').before(), n_lifts=1)
        diag_syrk_scheduled = simplify(diag_syrk_scheduled)
        diag_syrk_scheduled = reorder_loops(diag_syrk_scheduled, 'ii jo')
        diag_syrk_scheduled = replace(diag_syrk_scheduled, 'for ii in _:_ #0', gebp_diag_handler.base_gebp)
        diag_syrk_scheduled = call_eqv(diag_syrk_scheduled, f'gebp_base_{gebp_diag_handler.this_id}(_)', gebp_diag_handler.scheduled_gebp)
        print(diag_syrk_scheduled)

        microkernel_diag_handler = Microkernel(self.machine, self.microkernel.M_r, self.microkernel.N_r, self.K_blk, self.precision)
        diag_syrk_scheduled = divide_loop(diag_syrk_scheduled, 'for ii in _:_', microkernel_diag_handler.M_r, ['iio', 'iii'], tail='cut')
        diag_syrk_scheduled = divide_loop(diag_syrk_scheduled, 'for ji in _:_', microkernel_diag_handler.N_r, ['jio', 'jii'], tail='cut')
        diag_syrk_scheduled = autofission(diag_syrk_scheduled, diag_syrk_scheduled.find('for jii in _:_ #1').before(), n_lifts=1)
        diag_syrk_scheduled = simplify(diag_syrk_scheduled)
        diag_syrk_scheduled = reorder_loops(diag_syrk_scheduled, 'iii jio')
        diag_syrk_scheduled = replace(diag_syrk_scheduled, 'for iii in _:_ #0', microkernel_diag_handler.base_microkernel)
        diag_syrk_scheduled = call_eqv(diag_syrk_scheduled, f'microkernel_{microkernel_diag_handler.this_id}(_)', microkernel_diag_handler.scheduled_microkernel)
        print(diag_syrk_scheduled)

        gepp_syrk_scheduled = call_eqv(gepp_syrk_scheduled, 'diag_handler(_)', diag_syrk_scheduled)

        return gepp_syrk_scheduled, gepp_syrk_base

    
    def schedule_syrk_lower_notranspose(self, ssyrk_base: Procedure):
        syrk = divide_loop(ssyrk_base, 'k', self.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        syrk = autofission(syrk, syrk.find('for ko in _:_ #0').after(), n_lifts=2)
        syrk = reorder_loops(syrk, 'j ko')
        syrk = reorder_loops(syrk, 'i ko')
        syrk = replace(syrk, 'for i in _:_ #0', self.gepp_syrk_base_lower_notranspose)
        syrk = call_eqv(syrk, 'gepp_syrk_base(_)', self.gepp_syrk_scheduled_lower_notranspose)
        return syrk
    

    def specialize_syrk(self, syrk: Procedure, precision: str):
        prefix = "s" if precision == "f32" else "d"
        name = syrk.name().replace("exo_", "")
        syrk = rename(syrk, "exo_" + prefix + name)
        args = ["A1", "A2", "C", "alpha", "beta"]
        for arg in args:
            syrk = set_precision(syrk, arg, precision)
        return syrk

 
k_blk = C.syrk.k_blk
m_blk = C.syrk.m_blk
m_reg = C.syrk.m_reg
n_reg = C.syrk.n_reg


ssyrk = SYRK(C.Machine, 'f32',
             k_blk, m_blk, 
             m_reg, n_reg
             )

exo_ssyrk_lower_notranspose = ssyrk.entry_points[0]

C.Machine.vec_width //= 2
dsyrk = SYRK(C.Machine, 'f64',
             k_blk, m_blk,
             m_reg, n_reg//2
             )

exo_dsyrk_lower_notranspose = dsyrk.entry_points[0]

__all__ = [p.name() for p in ssyrk.entry_points] + [p.name() for p in dsyrk.entry_points]
