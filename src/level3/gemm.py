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
        @proc
        def gemm_apply_scalar(
            M: size, 
            N: size, 
            scalar: f32[1], 
            A: f32[M, N] @ DRAM
        ):
            for i in seq(0, M):
                for j in seq(0, N):
                    A[i, j] = A[i, j] * scalar[0]

        @proc
        def do_nothing():
            pass                   

        ### Process format options

        if transb==ExoBlasNoTranspose:
            if transa==ExoBlasNoTranspose:
                self.gemm_base = gemm_notranspose
            elif transa==ExoBlasTranspose:
                raise Exception("Not implemented")
        elif transb==ExoBlasTranspose:
            if transa==ExoBlasNoTranspose:
                raise Exception("Not implemented")
            elif transa==ExoBlasTranspose:
                raise Exception("Not implemented")
        
        if alpha==1.0:
            apply_alpha = do_nothing
        else:
            apply_alpha = self.schedule_apply_scalar(gemm_apply_scalar, machine, 'A')
            
        
        if beta==1.0:
            apply_beta = do_nothing
        else:
            apply_beta = self.schedule_apply_scalar(gemm_apply_scalar, machine, 'B')
        

        ### Schedule GEMM
        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk) #TODO: Add precision args to microkernel, gebp, gepp
        self.gebp = GEBP_kernel(self.microkernel, M_blk)
        self.gepp = GEPP_kernel(self.gebp)

        gemm_scheduled = self.schedule_gemm_notranspose()


        ### Create final GEMM
        @proc
        def exo_gemm(
            M: size,
            N: size,
            K: size,
            alpha: f32[1],
            beta: f32[1],
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            apply_alpha(M, K, alpha, A)
            apply_beta(M, N, beta, C)
            gemm_scheduled(M, N, K, C, A, B)
        
        self.exo_gemm = exo_gemm


    def schedule_gemm_notranspose(self):
        gemm_scheduled = divide_loop(self.gemm_base, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')
        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)
        return gemm_scheduled


    def bind(self, proc, buffer, reg, machine):
        proc = bind_expr(proc, buffer, reg)
        proc = expand_dim(proc, reg, machine.vec_width, "ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg} = _").after())
        return proc    


    def stage(self, proc, buffer, reg, machine):
        proc = stage_mem(proc, f'{buffer}[_] = _', f'{buffer}[i, ji]', reg)
        proc = expand_dim(proc, reg, machine.vec_width, "ji")
        proc = lift_alloc(proc, f"{reg} : _", n_lifts=2)
        proc = fission(proc, proc.find(f"{reg}[_] = _").after())
        return proc


    def schedule_apply_scalar(self, proc: Procedure, machine: "MachineParameters", buffer_name: str):
        
        proc = divide_loop(proc, 'j', machine.vec_width, ['jo', 'ji'], tail='cut_and_guard')
        proc = self.stage(proc, f'{buffer_name}', f'{buffer_name}_vec', machine)
        proc = self.bind(proc, 'scalar[_]', 'scalar_vec', machine)
        proc = fission(proc, proc.find(f'{buffer_name}_vec[_] = _ #1').after())
        proc = set_memory(proc, f'{buffer_name}_vec', machine.mem_type)
        proc = set_memory(proc, 'scalar_vec', machine.mem_type)

        print(proc)
        

        instr_lst = [machine.load_instr_f32, machine.broadcast_instr_f32, 
                                  machine.mul_instr_f32, machine.store_instr_f32]
        for instr in instr_lst:
            proc = replace_all(proc, instr)

        print(proc)

        return simplify(proc)


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

gemm = gemm.exo_gemm

__all__ = ['gemm']
