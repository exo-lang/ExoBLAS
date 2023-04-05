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
from format_options import *

import exo_blas_config as C

"""
TODO:
    - transpose
    - optimize alpha version
    - edge cases
"""

class GEMM:

    def __init__(self, machine: "MachineParameters",
                 precision: str,
                 K_blk: int, M_blk: int, 
                 M_reg: int, N_reg: int,
                 do_rename: bool = False,
                 main: bool = True):

        ### Specialize for different precisions
        self.precision = precision
        self.prefix = 's' if precision=='f32' else 'd'
        self.main = main
        

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
        gemm_notranspose_noalpha = rename(gemm_notranspose_noalpha, f'{self.prefix}{gemm_notranspose_noalpha.name()}_{M_blk}_{K_blk}')
        gemm_notranspose_noalpha = self.specialize_gemm(gemm_notranspose_noalpha, self.precision, ["A", "B", "C"])

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
        gemm_notranspose_alpha = rename(gemm_notranspose_alpha, f'{self.prefix}{gemm_notranspose_alpha.name()}_{M_blk}_{K_blk}')
        gemm_notranspose_alpha = self.specialize_gemm(gemm_notranspose_alpha, self.precision, ["A", "B", "C", "alpha", "temp"])

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
        gemm_apply_scalar = set_precision(gemm_apply_scalar, "scalar", self.precision)
        gemm_apply_scalar = set_precision(gemm_apply_scalar, "P", self.precision)

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
        gemm_apply_scalar_no_overwrite = set_precision(gemm_apply_scalar_no_overwrite, "scalar", self.precision)
        gemm_apply_scalar_no_overwrite = set_precision(gemm_apply_scalar_no_overwrite, "P", self.precision)    
        gemm_apply_scalar_no_overwrite = set_precision(gemm_apply_scalar_no_overwrite, "Q", self.precision)                               
        
        ### Alpha and Beta scaling procedures
        apply_alpha = self.schedule_apply_scalar(gemm_apply_scalar_no_overwrite, machine, ['Q', 'P'], f'{self.prefix}_apply_alpha_{M_blk}_{K_blk}', False)
        apply_beta = self.schedule_apply_scalar(gemm_apply_scalar, machine, ['P'], f'{self.prefix}_apply_beta_{M_blk}_{K_blk}', True)

        ### GEMM kernels
        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk, self.precision) #TODO: Add precision args to microkernel, gebp, gepp
        self.gebp = GEBP_kernel(self.microkernel, M_blk, self.precision)
        self.gepp = GEPP_kernel(self.gebp, self.precision)
        
        #self.microkernel.base_microkernel = self.specialize_gemm(self.microkernel.base_microkernel, precision, ["A", "B", "C"], False)
        #self.microkernel.scheduled_microkernel = self.specialize_gemm(self.microkernel.scheduled_microkernel, precision, ["A", "B", "C"], False)
        #self.gebp.base_gebp = self.specialize_gemm(self.gebp.base_gebp, precision, ["A", "B", "C"], False)
        #self.gebp.scheduled_gebp = self.specialize_gemm(self.gebp.scheduled_gebp, precision, ["A", "B", "C"], False)
        #self.gepp.gepp_base = self.specialize_gemm(self.gepp.gepp_base, precision, ["A", "B", "C"], False)
        #self.gepp.gepp_scheduled = self.specialize_gemm(self.gepp.gepp_scheduled, precision, ["A", "B", "C"], False)

        ### GEMM variants
        gemm_scheduled_notranspose_noalpha = self.schedule_gemm_notranspose_noalpha(gemm_notranspose_noalpha)
        gemm_scheduled_notranspose_alpha = self.schedule_gemm_notranspose_alpha(gemm_notranspose_alpha, gemm_apply_scalar_no_overwrite, apply_alpha)
        #gemm_scheduled_transpose_noalpha
        #gemm_scheduled_transpose_alpha

        ### Create entry points

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
        exo_gemm_notranspose_noalpha_nobeta = self.specialize_gemm(exo_gemm_notranspose_noalpha_nobeta, self.precision)
        print(gemm_scheduled_notranspose_noalpha)
        
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
        exo_gemm_alphazero_nobeta = self.specialize_gemm(exo_gemm_alphazero_nobeta, self.precision)

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
        exo_gemm_alphazero_beta = self.specialize_gemm(exo_gemm_alphazero_beta, self.precision)
        
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
        exo_gemm_notranspose_alpha_nobeta = self.specialize_gemm(exo_gemm_notranspose_alpha_nobeta, self.precision)
        
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
        exo_gemm_notranspose_alpha_beta = self.specialize_gemm(exo_gemm_notranspose_alpha_beta, self.precision)
        
        self.entry_points = [
            exo_gemm_notranspose_noalpha_nobeta,
            exo_gemm_alphazero_nobeta,
            exo_gemm_alphazero_beta,
            exo_gemm_notranspose_alpha_nobeta,
            exo_gemm_notranspose_alpha_beta
        ]
        
        print(self.microkernel.scheduled_microkernel)
        print(apply_beta)
        print(apply_alpha)
        #print(self.gebp.scheduled_gebp)
        #print(self.gepp.gepp_scheduled)

        if do_rename:
            for i in range(len(self.entry_points)):
                self.entry_points[i] = rename(self.entry_points[i], f'{self.entry_points[i].name()}_{M_blk}_{K_blk}')


    def schedule_gemm_notranspose_noalpha(self, gemm_procedure: Procedure):
        gemm_scheduled = divide_loop(gemm_procedure, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')
        #gemm_scheduled = stage_mem(gemm_scheduled, 'for i in _:_ #0', f'B[{self.gepp.K_blk} * ko:{self.gepp.K_blk} + {self.gepp.K_blk} * ko, 0:N]', 'B_packed')
        if self.gepp.K_blk>256:
            gemm_scheduled = stage_mem(gemm_scheduled, 'for i in _:_ #0', f'A[0:M, {self.gepp.K_blk} * ko:{self.gepp.K_blk} + {self.gepp.K_blk} * ko]', 'A_packed')
        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, f'gepp_base_{self.gepp.this_id}(_)', self.gepp.gepp_scheduled)
        return simplify(gemm_scheduled)


    def schedule_gemm_notranspose_alpha(self, gemm_procedure: Procedure, apply_alpha_base: Procedure, apply_alpha_scheduled: Procedure):

        gemm_scheduled = divide_loop(gemm_procedure, 'k #1', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        gemm_scheduled = autofission(gemm_scheduled, gemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        gemm_scheduled = reorder_loops(gemm_scheduled, 'j ko')
        gemm_scheduled = reorder_loops(gemm_scheduled, 'i ko')

        gemm_scheduled = replace(gemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        gemm_scheduled = call_eqv(gemm_scheduled, f'gepp_base_{self.gepp.this_id}(_)', self.gepp.gepp_scheduled)

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
        for buffer in buffer_names + ['scalar']:
            proc = set_precision(proc, buffer, self.precision)

        proc = divide_loop(proc, 'j', machine.vec_width, ['jo', 'ji'], tail='cut_and_guard')
        proc = self.bind(proc, 'scalar[_]', 'scalar_vec', machine)
        if len(buffer_names)>1:
            proc = self.bind(proc, f'{buffer_names[1]}[_]', f'{buffer_names[1]}_vec', machine)
            proc = set_precision(proc, f'{buffer_names[1]}_vec', self.precision)
        proc = self.stage(proc, f'{buffer_names[0]}', f'{buffer_names[0]}_vec', machine)
    
        if apply_hack:
            proc = fission(proc, proc.find(f'{buffer_names[0]}_vec[_] = _ #1').after())  

        for buffer_name in buffer_names:       
            proc = set_memory(proc, f'{buffer_name}_vec', machine.mem_type)
        proc = set_memory(proc, 'scalar_vec', machine.mem_type)
        proc = set_precision(proc, 'scalar_vec', self.precision)
        
        if self.precision=='f32':
            instr_lst = [machine.load_instr_f32, machine.broadcast_instr_f32,
                                  machine.reg_copy_instr_f32,
                                  machine.mul_instr_f32, machine.store_instr_f32]
            mul_hack_instr = machine.mul_instr_f32_hack
        else:
            instr_lst = [machine.load_instr_f64, machine.broadcast_instr_f64,
                                  machine.reg_copy_instr_f64,
                                  machine.mul_instr_f64, machine.store_instr_f64]
            mul_hack_instr = machine.mul_instr_f64_hack

        if apply_hack:
            instr_lst[3] = mul_hack_instr

        for instr in instr_lst:
            proc = replace_all(proc, instr)
        
        if self.main:
            proc = rename(proc, proc.name() + "_main")

        return simplify(proc)
    

    def specialize_gemm(self, gemm: Procedure, precision: str, args: list[str]=["A", "B", "C", "alpha", "beta"]):
        
        prefix = "s" if precision == "f32" else "d"
        name = gemm.name().replace("exo_", "")
        specialized = rename(gemm, "exo_" + prefix + name)
            
        for arg in args:
            specialized = set_precision(specialized, arg, precision)
        
        if self.main:
            specialized = rename(specialized, specialized.name() + "_main")

        return specialized


k_blk = C.gemm.k_blk
m_blk = C.gemm.m_blk
m_reg = C.gemm.m_reg
n_reg = C.gemm.n_reg

#################################################
# Generate f32 kernels
#################################################

sgemm_main = GEMM(
    C.Machine,
    'f32', 
    k_blk, m_blk, 
    m_reg, n_reg,
)

blk_sizes = [2**i for i in range(5, 9)]
sgemm_backup_kernels = [GEMM(C.Machine, 'f32', blk, blk, m_reg, n_reg, True, False) for blk in blk_sizes] # Use these if problem size is too small for the main block size

exo_sgemm_notranspose_noalpha_nobeta_32_32 = sgemm_backup_kernels[0].entry_points[0]
exo_sgemm_alphazero_nobeta_32_32 = sgemm_backup_kernels[0].entry_points[1]
exo_sgemm_alphazero_beta_32_32 = sgemm_backup_kernels[0].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_32_32 = sgemm_backup_kernels[0].entry_points[3]
exo_sgemm_notranspose_alpha_beta_32_32 = sgemm_backup_kernels[0].entry_points[4]

exo_sgemm_notranspose_noalpha_nobeta_64_64 = sgemm_backup_kernels[1].entry_points[0]
exo_sgemm_alphazero_nobeta_64_64 = sgemm_backup_kernels[1].entry_points[1]
exo_sgemm_alphazero_beta_64_64 = sgemm_backup_kernels[1].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_64_64 = sgemm_backup_kernels[1].entry_points[3]
exo_sgemm_notranspose_alpha_beta_64_64 = sgemm_backup_kernels[1].entry_points[4]

exo_sgemm_notranspose_noalpha_nobeta_128_128 = sgemm_backup_kernels[2].entry_points[0]
exo_sgemm_alphazero_nobeta_128_128 = sgemm_backup_kernels[2].entry_points[1]
exo_sgemm_alphazero_beta_128_128 = sgemm_backup_kernels[2].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_128_128 = sgemm_backup_kernels[2].entry_points[3]
exo_sgemm_notranspose_alpha_beta_128_128 = sgemm_backup_kernels[2].entry_points[4]

exo_sgemm_notranspose_noalpha_nobeta_256_256 = sgemm_backup_kernels[3].entry_points[0]
exo_sgemm_alphazero_nobeta_256_256 = sgemm_backup_kernels[3].entry_points[1]
exo_sgemm_alphazero_beta_256_256 = sgemm_backup_kernels[3].entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_256_256 = sgemm_backup_kernels[3].entry_points[3]
exo_sgemm_notranspose_alpha_beta_256_256 = sgemm_backup_kernels[3].entry_points[4]

exo_sgemm_notranspose_noalpha_nobeta_main = sgemm_main.entry_points[0]
exo_sgemm_alphazero_nobeta_main = sgemm_main.entry_points[1]
exo_sgemm_alphazero_beta_main = sgemm_main.entry_points[2]
exo_sgemm_notranspose_alpha_nobeta_main = sgemm_main.entry_points[3]
exo_sgemm_notranspose_alpha_beta_main = sgemm_main.entry_points[4]

sgemm_backup_entry_points = []
for kernel in sgemm_backup_kernels:
    sgemm_backup_entry_points.extend(kernel.entry_points)

sgemm_entry_points = [p.name() for p in sgemm_main.entry_points] + [p.name() for p in sgemm_backup_entry_points]

#################################################
# Generate f64 kernels
#################################################

C.Machine.vec_width //= 2
print(C.Machine.vec_width)

dgemm_main = GEMM(
    C.Machine,
    'f64', 
    k_blk, m_blk, 
    m_reg, n_reg//2
)

dgemm_backup_kernels = [GEMM(C.Machine, 'f64', blk, blk, m_reg, n_reg//2, True, False) for blk in blk_sizes] # Use these if problem size is too small for the main block size

exo_dgemm_notranspose_noalpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[0]
exo_dgemm_alphazero_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[1]
exo_dgemm_alphazero_beta_32_32 = dgemm_backup_kernels[0].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_32_32 = dgemm_backup_kernels[0].entry_points[3]
exo_dgemm_notranspose_alpha_beta_32_32 = dgemm_backup_kernels[0].entry_points[4]

exo_dgemm_notranspose_noalpha_nobeta_64_64 = dgemm_backup_kernels[1].entry_points[0]
exo_dgemm_alphazero_nobeta_64_64 = dgemm_backup_kernels[1].entry_points[1]
exo_dgemm_alphazero_beta_64_64 = dgemm_backup_kernels[1].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_64_64 = dgemm_backup_kernels[1].entry_points[3]
exo_dgemm_notranspose_alpha_beta_64_64 = dgemm_backup_kernels[1].entry_points[4]

exo_dgemm_notranspose_noalpha_nobeta_128_128 = dgemm_backup_kernels[2].entry_points[0]
exo_dgemm_alphazero_nobeta_128_128 = dgemm_backup_kernels[2].entry_points[1]
exo_dgemm_alphazero_beta_128_128 = dgemm_backup_kernels[2].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_128_128 = dgemm_backup_kernels[2].entry_points[3]
exo_dgemm_notranspose_alpha_beta_128_128 = dgemm_backup_kernels[2].entry_points[4]

exo_dgemm_notranspose_noalpha_nobeta_256_256 = dgemm_backup_kernels[3].entry_points[0]
exo_dgemm_alphazero_nobeta_256_256 = dgemm_backup_kernels[3].entry_points[1]
exo_dgemm_alphazero_beta_256_256 = dgemm_backup_kernels[3].entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_256_256 = dgemm_backup_kernels[3].entry_points[3]
exo_dgemm_notranspose_alpha_beta_256_256 = dgemm_backup_kernels[3].entry_points[4]

exo_dgemm_notranspose_noalpha_nobeta_main = dgemm_main.entry_points[0]
exo_dgemm_alphazero_nobeta_main = dgemm_main.entry_points[1]
exo_dgemm_alphazero_beta_main = dgemm_main.entry_points[2]
exo_dgemm_notranspose_alpha_nobeta_main = dgemm_main.entry_points[3]
exo_dgemm_notranspose_alpha_beta_main = dgemm_main.entry_points[4]


dgemm_backup_entry_points = []
for kernel in dgemm_backup_kernels:
    dgemm_backup_entry_points.extend(kernel.entry_points)

dgemm_entry_points = [p.name() for p in dgemm_main.entry_points] + [p.name() for p in dgemm_backup_entry_points]

__all__ = sgemm_entry_points + dgemm_entry_points
print(__all__)
