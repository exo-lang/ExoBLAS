from __future__ import annotations

from email.mime import base
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

class Microkernel:

    microkernel_id = 0

    def __init__(self, machine: MachineParameters, M_r: int, N_r: int, K_blk: int):
        
        # Problem dimensions
        self.M_r = M_r
        self.N_r = N_r
        self.K_blk = K_blk
        self.this_id = Microkernel.microkernel_id

        # Base SGEMM procedure
        @proc
        def SGEMM(
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

        self.sgemm_window = (rename(SGEMM, 'sgemm_win'))
        self.sgemm_window = set_window(self.sgemm_window, 'A', True)
        self.sgemm_window = set_window(self.sgemm_window, 'B', True)
        self.sgemm_window = set_window(self.sgemm_window, 'C', True)

        self.sgemm_base = SGEMM
        self.scheduled_microkernel, self.base_microkernel = self.generate_microkernel(machine, M_r, N_r, K_blk)

    def generate_microkernel(self, machine: MachineParameters, M_r: int, N_r: int, K_blk: int):
        """
        Generate microkernel for MxN problem size for the given machine.
        Vectorizes along dimension N or dimension K.
        """

        scheduled_microkernel, microkernel = self.do_generate_microkernel(machine, M_r, N_r, K_blk)

        Microkernel.microkernel_id += 1 # Ensure each microkernel has a unique name

        return scheduled_microkernel, microkernel

    
    def generate_base_microkernel(self, M_r: int, N_r: int, K_blk: int):

        microkernel = rename(self.sgemm_window, f"microkernel_{self.microkernel_id}")
        microkernel = microkernel.partial_eval(M=M_r, N=N_r, K=K_blk)
        return microkernel


    def do_generate_microkernel(self, machine: MachineParameters, M_r: int, N_r: int, K_blk: int):
        """
        Vectorize along dimension N
        """
        microkernel = self.generate_base_microkernel(M_r, N_r, K_blk)

        # Reorder and divide loops
        scheduled_microkernel = rename(microkernel, f"{machine.name}_microkernel_{M_r}x{N_r}_{self.microkernel_id}")
        scheduled_microkernel = reorder_loops(scheduled_microkernel, 'j k')
        scheduled_microkernel = reorder_loops(scheduled_microkernel, 'i k')
        scheduled_microkernel = divide_loop(scheduled_microkernel, 'j', machine.vec_width, ['jo', 'ji'], perfect=True)
        
        # Create C buffer in vector mem
        c_reg_str = f'C[i, {machine.vec_width} * jo + ji]'
        scheduled_microkernel = stage_mem(scheduled_microkernel, 'C[_] += _', c_reg_str, 'C_reg')
        scheduled_microkernel = set_memory(scheduled_microkernel, 'C_reg', machine.mem_type)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'C_reg', machine.vec_width, 'ji', unsafe_disable_checks=True)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'C_reg', N_r//machine.vec_width, f'jo', unsafe_disable_checks=True)

        scheduled_microkernel = expand_dim(scheduled_microkernel, 'C_reg', M_r, 'i', unsafe_disable_checks=True)
        scheduled_microkernel = lift_alloc(scheduled_microkernel, 'C_reg', n_lifts=4)
        scheduled_microkernel = autofission(scheduled_microkernel, scheduled_microkernel.find('C_reg[_] = _').after(), n_lifts=4)
        scheduled_microkernel = autofission(scheduled_microkernel, scheduled_microkernel.find('C[_] = _').before(), n_lifts=4)

        # Setup A buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, 'A[_]', 'A_vec')
        scheduled_microkernel = set_memory(scheduled_microkernel, 'A_vec', machine.mem_type)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'A_vec', machine.vec_width, 'ji', unsafe_disable_checks=True)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'A_vec', M_r, 'i', unsafe_disable_checks=True)

        # Setup B buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, 'B[_]', 'B_vec')

        scheduled_microkernel = set_memory(scheduled_microkernel, 'B_vec', machine.mem_type)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'B_vec', machine.vec_width, f'ji', unsafe_disable_checks=True)
        scheduled_microkernel = expand_dim(scheduled_microkernel, 'B_vec', (N_r // machine.vec_width), f'jo', unsafe_disable_checks=True)

        # Move A_vec and B_vec into proper sites
        scheduled_microkernel = lift_alloc(scheduled_microkernel, 'A_vec', n_lifts=3)
        scheduled_microkernel = autofission(scheduled_microkernel, scheduled_microkernel.find('A_vec[_] = _').after(), n_lifts=3)
        scheduled_microkernel = lift_alloc(scheduled_microkernel, 'B_vec', n_lifts=3)
        scheduled_microkernel = autofission(scheduled_microkernel, scheduled_microkernel.find('B_vec[_] = _').after(), n_lifts=3)

        # Replace
        scheduled_microkernel = replace_all(scheduled_microkernel, machine.load_instr)
        scheduled_microkernel = replace_all(scheduled_microkernel, machine.broadcast_instr)
        scheduled_microkernel = replace_all(scheduled_microkernel, machine.store_instr)
        scheduled_microkernel = replace_all(scheduled_microkernel, machine.fmadd_instr)
        scheduled_microkernel = simplify(scheduled_microkernel)

        return scheduled_microkernel, microkernel

#test_microkernel = Microkernel(NeonMachine, 4, 16, 32)
#print("TEST PASSED: Microkernel successfully generated!")


"""
TODO: Generate transpose version of all of these 
"""
class GEBP_kernel:

    gebp_id = 0

    def __init__(self, microkernel: Microkernel, 
                 M_blk: int):

        # Problem dimensions
        self.M_blk = M_blk
        self.microkernel = microkernel
        self.this_id = GEBP_kernel.gebp_id
        
        # Base SGEMM procedure
        @proc
        def SGEMM(
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

        self.sgemm_window = (rename(SGEMM, 'sgemm_win'))
        self.sgemm_window = set_window(self.sgemm_window, 'A', True)
        self.sgemm_window = set_window(self.sgemm_window, 'B', True)
        self.sgemm_window = set_window(self.sgemm_window, 'C', True)

        self.sgemm_base = SGEMM

        self.scheduled_gebp, self.base_gebp = self.generate_gebp()


    def generate_base_gebp(self, M_blk: int, K_blk: int):
        base_microkernel = rename(self.sgemm_window, f"gebp_base_{self.this_id}")
        return simplify(base_microkernel.partial_eval(M=M_blk, K=K_blk)) 
    

    def generate_gebp(self):
        
        scheduled_gebp, gebp = self.do_generate_gebp()

        GEBP_kernel.gebp_id += 1

        return scheduled_gebp, gebp

    def do_generate_gebp(self):

        gebp = self.generate_base_gebp(self.M_blk, self.microkernel.K_blk)

        scheduled_gebp = rename(gebp, f"gebp_{self.M_blk}x{self.microkernel.K_blk}_{self.gebp_id}")
        scheduled_gebp = divide_loop(scheduled_gebp, 'i', self.microkernel.M_r, ['io', 'ii'], tail='cut_and_guard')
        scheduled_gebp = simplify(divide_loop(scheduled_gebp, 'j', self.microkernel.N_r, ['jo', 'ji'], tail='cut_and_guard'))
        
        scheduled_gebp = autofission(scheduled_gebp, scheduled_gebp.find('for jo in _: _').after(), n_lifts=2)
        scheduled_gebp = reorder_loops(scheduled_gebp, 'ii jo')
        scheduled_gebp = replace_all(scheduled_gebp, self.microkernel.base_microkernel)
        scheduled_gebp = call_eqv(scheduled_gebp, f'microkernel_{self.microkernel.this_id}(_)', self.microkernel.scheduled_microkernel)

        scheduled_gebp = reorder_loops(scheduled_gebp, 'io jo')
        scheduled_gebp = stage_mem(scheduled_gebp, 'for io in _:_ #0', f'B[0:{self.microkernel.K_blk}, {self.microkernel.N_r}*jo:{self.microkernel.N_r}*jo+{self.microkernel.N_r}]', 'B_strip')
        
        return simplify(scheduled_gebp), gebp


class GEPP_kernel:

    def __init__(self, gebp: GEBP_kernel):

        self.K_blk = gebp.microkernel.K_blk
        self.gebp = gebp

        # Base SGEMM procedure
        @proc
        def SGEMM(
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

        self.sgemm_window = (rename(SGEMM, 'sgemm_win'))
        self.sgemm_window = set_window(self.sgemm_window, 'A', True)
        self.sgemm_window = set_window(self.sgemm_window, 'B', True)
        self.sgemm_window = set_window(self.sgemm_window, 'C', True)

        self.sgemm_base = SGEMM

        self.gepp_base, self.gepp_scheduled = self.generate_gepp()

    
    def generate_base_gepp(self):
        base_gepp = rename(self.sgemm_window, "gepp_base")
        return base_gepp.partial_eval(K=self.K_blk)


    def generate_gepp(self):
        base_gepp, scheduled_gepp = self.do_generate_gepp()
        return base_gepp, scheduled_gepp

    def do_generate_gepp(self):

        base_gepp = self.generate_base_gepp()

        scheduled_gepp = rename(base_gepp, "scheduled_gepp")
        scheduled_gepp = divide_loop(scheduled_gepp, 'i', self.gebp.M_blk, ['io', 'ii'], tail='cut_and_guard')
        scheduled_gepp = replace(scheduled_gepp, 'for ii in _:_ #0', self.gebp.base_gebp)
        scheduled_gepp = call_eqv(scheduled_gepp, f'gebp_base_{self.gebp.this_id}(_)', self.gebp.scheduled_gebp)
        return base_gepp, scheduled_gepp

#test_gebp = GEBP_kernel(test_microkernel, 64, 256)
#print("TEST PASSED: GEBP kernel successfully generated!")
