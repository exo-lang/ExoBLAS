from __future__ import annotations
from importlib.metadata import entry_points
from os import abort

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel
from format_options import *

import exo_blas_config as C

"""
TODO:
    1. Variable block sizes based on problem size
    2. transpose gemm
    3. dgemm
"""

class GEMM:

    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 K_blk: int, M_blk: int, 
                 M_reg: int, N_reg: int):

        ### GEMM PROCEDURES
        @proc
        def gemm_notranspose_noalpha(
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
        def gemm_notranspose_alpha(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            temp: f32[K, N]
            for j in seq(0, N):
                for k in seq(0, K):
                    temp[k, j] = B[k, j] * alpha[0]
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i,k] * temp[k, j]   


        ### ALPHA AND BETA
        @proc
        def gemm_apply_scalar(
            M: size, 
            N: size, 
            scalar: f32[1], 
            P: f32[M, N] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    P[i, j] = P[i, j] * scalar[0]

        @proc
        def gemm_apply_scalar_no_overwrite(
            M: size, 
            N: size, 
            scalar: f32[1], 
            P: f32[M, N] @ DRAM,
            Q: f32[M, N] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    Q[i, j] = P[i, j] * scalar[0]                    


        @proc
        def do_nothing():
            pass                   
            
        
        ### Alpha and Beta scaling procedures
        apply_alpha = self.schedule_apply_scalar(gemm_apply_scalar_no_overwrite, machine, ['Q', 'P'], 'apply_alpha', False)
        apply_beta = self.schedule_apply_scalar(gemm_apply_scalar, machine, ['P'], 'apply_beta', True)
        

        ### GEMM kernels
        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk) #TODO: Add precision args to microkernel, gebp, gepp
        self.gebp = GEBP_kernel(self.microkernel, M_blk)
        self.gepp = GEPP_kernel(self.gebp)
        

        ### GEMM variants
        gemm_scheduled_notranspose_noalpha = self.schedule_gemm_notranspose_noalpha(gemm_notranspose_noalpha)
        gemm_scheduled_notranspose_alpha = self.schedule_gemm_notranspose_alpha(gemm_notranspose_alpha, gemm_apply_scalar_no_overwrite, apply_alpha)


        ### Create final GEMM procedures

        # Alpha = Beta = 1
        @proc
        def exo_gemm_notranspose_noalpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM
        ):
            gemm_scheduled_notranspose_noalpha(M, N, K, C, A, B)
        
        # Alpha = 0, Beta = 0
        @proc
        def exo_gemm_alphazero_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM
        ):
            pass

        # Alpha = 0, Beta != 0
        @proc
        def exo_gemm_alphazero_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM
        ):
            apply_beta(M, N, beta, C)
        
        # Alpha != 1, Beta = 1
        @proc
        def exo_gemm_notranspose_alpha_nobeta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM
        ):
            gemm_scheduled_notranspose_alpha(M, N, K, alpha, C, A, B)
        
        # Alpha = Beta != 1
        @proc
        def exo_gemm_notranspose_alpha_beta(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
            C: f32[M, N] @ DRAM
        ):
            apply_beta(M, N, beta, C)
            gemm_scheduled_notranspose_alpha(M, N, K, alpha, C, A, B)
        
        self.entry_points = [
            exo_gemm_notranspose_noalpha_nobeta,
            exo_gemm_alphazero_nobeta,
            exo_gemm_alphazero_beta,
            exo_gemm_notranspose_alpha_nobeta,
            exo_gemm_notranspose_alpha_beta
        ]


    def schedule_gemm_notranspose_noalpha(self, gemm_procedure: Procedure):
        gemm_scheduled = divide_loop(gemm_procedure, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')
        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)
        return simplify(gemm_scheduled)


    def schedule_gemm_notranspose_alpha(self, gemm_procedure: Procedure, apply_alpha_base: Procedure, apply_alpha_scheduled: Procedure):

        gemm_scheduled = divide_loop(gemm_procedure, 'k #1', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')

        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)

        gemm_scheduled = replace(gemm_scheduled, 'for j in _:_ #0', reorder_loops(apply_alpha_base, 'i j'))
        gemm_scheduled = call_eqv(gemm_scheduled, 'gemm_apply_scalar_no_overwrite(_)', apply_alpha_scheduled)


        gemm_scheduled = simplify(gemm_scheduled)

        return gemm_scheduled


    def bind(self, proc, buffer, reg, machine):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, machine.vec_width, "ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg} = _").after())
        return proc    


    def stage(self, proc, buffer, reg, machine):
        proc = stage_mem(proc, f'{buffer}[_] = _', f'{buffer}[i, ji + {machine.vec_width}*jo]', reg)
        proc = expand_dim(proc, reg, machine.vec_width, f"ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc


    def schedule_apply_scalar(self, proc: Procedure, machine: "MachineParameters", 
                              buffer_names: list, name: str, apply_hack: bool):
        
        proc = rename(proc, name)
        proc = divide_loop(proc, 'j', machine.vec_width, ['jo', 'ji'], tail='cut_and_guard')
        proc = self.bind(proc, 'scalar[_]', 'scalar_vec', machine)
        if len(buffer_names)>1:
            proc = self.bind(proc, f'{buffer_names[1]}[_]', f'{buffer_names[1]}_vec', machine)
        proc = self.stage(proc, f'{buffer_names[0]}', f'{buffer_names[0]}_vec', machine)
    
        if apply_hack:
            proc = fission(proc, proc.find(f'{buffer_names[0]}_vec[_] = _ #1').after())  

        for buffer_name in buffer_names:       
            proc = set_memory(proc, f'{buffer_name}_vec', machine.mem_type)
        proc = set_memory(proc, 'scalar_vec', machine.mem_type)

        if apply_hack:
            instr_lst = [machine.load_instr_f32, machine.broadcast_instr_f32,
                                machine.reg_copy_instr_f32,
                                machine.mul_instr_f32_hack, machine.store_instr_f32]
        else:
            instr_lst = [machine.load_instr_f32, machine.broadcast_instr_f32,
                                  machine.reg_copy_instr_f32,
                                  machine.mul_instr_f32, machine.store_instr_f32]
        for instr in instr_lst:
            proc = replace_all(proc, instr)

        return simplify(proc)


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg


gemm = GEMM(
    C.Machine,
    'f32', 
    k_blk, m_blk, 
    m_reg, n_reg
)

exo_gemm_notranspose_noalpha_nobeta = gemm.entry_points[0]
exo_gemm_alphazero_nobeta = gemm.entry_points[1]
exo_gemm_alphazero_beta = gemm.entry_points[2]
exo_gemm_notranspose_alpha_nobeta = gemm.entry_points[3]
exo_gemm_notranspose_alpha_beta = gemm.entry_points[4]

__all__ = [p.name() for p in gemm.entry_points]
