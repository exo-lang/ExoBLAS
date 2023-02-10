from __future__ import annotations
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEBP_kernel, Microkernel, NeonMachine

class SYRK:

    def __init__(self, gebp_kernel: GEBP_kernel):

        # Problem dimensions
        self.gebp_kernel = gebp_kernel
        self.microkernel = gebp_kernel.microkernel
        self.K_blk = self.microkernel.K_blk
        self.M_blk = gebp_kernel.M_blk

        @proc
        def SYRK(N: size, 
                 K: size, 
                 A: f32[N, K] @ DRAM, 
                 A_t: f32[K, N] @ DRAM,
                 C: f32[N, N] @ DRAM):
            assert N >= 1
            assert K >= 1
            assert stride(A, 1) == 1
            assert stride(A_t, 1) == 1
            assert stride(C, 1) == 1

            for i in seq(0, N):
                for j in seq(0, i+1):
                    for k in seq(0, K):
                        C[i, j] += A[i, k]*A_t[k, j]

        self.syrk_base = SYRK

        self.syrk_win = rename(self.syrk_base, "syrk_win")
        self.syrk_win = set_window(self.syrk_win, 'A', True)
        self.syrk_win = set_window(self.syrk_win, 'A_t', True)
        self.syrk_win = set_window(self.syrk_win, 'C', True)

        self.gepp_syrk_scheduled, self.gepp_syrk_base = self.generate_syrk_gepp()
        self.syrk_scheduled = self.schedule_gepp()
        print(self.gebp_kernel.base_gebp)



    
    def generate_gepp_syrk_base(self):
        gepp_syrk_base = rename(self.syrk_win, "gepp_syrk_base")
        gepp_syrk_base = gepp_syrk_base.partial_eval(K=self.microkernel.K_blk)
        return gepp_syrk_base
    

    def generate_syrk_gepp(self):

        gepp_syrk_scheduled, gepp_syrk_base = self.do_generate_syrk_gepp()

        return gepp_syrk_scheduled, gepp_syrk_base
    

    def do_generate_syrk_gepp(self):

        gepp_syrk_base = self.generate_gepp_syrk_base()

        gepp_syrk_scheduled = rename(gepp_syrk_base, "gepp_syrk_scheduled")
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'i', self.M_blk, ['io', 'ii'], tail='cut_and_guard')
        gepp_syrk_scheduled = cut_loop(gepp_syrk_scheduled, 'for j in _:_', 1)
        gepp_syrk_scheduled = divide_loop(gepp_syrk_scheduled, 'j #1', self.M_blk, ['jo', 'ji'], tail='cut_and_guard')
        print(gepp_syrk_scheduled)

        gepp_syrk_scheduled = reorder_stmts(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_ #0').expand(1))
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').after(), n_lifts=1)
        gepp_syrk_scheduled = autofission(gepp_syrk_scheduled, gepp_syrk_scheduled.find('for j in _:_').before(), n_lifts=1)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)
        print(gepp_syrk_scheduled )

        gepp_syrk_scheduled = reorder_loops(gepp_syrk_scheduled, 'ii jo')
        gepp_syrk_scheduled = replace(gepp_syrk_scheduled, 'for ii in _:_ #0', gebp.base_gebp)
        gepp_syrk_scheduled = call_eqv(gepp_syrk_scheduled, f'gebp_base_{gebp.this_id}(_)', gebp.scheduled_gebp)
        gepp_syrk_scheduled = simplify(gepp_syrk_scheduled)
        print(gepp_syrk_scheduled )

        return gepp_syrk_scheduled, gepp_syrk_base

    
    def schedule_gepp(self):

        syrk = divide_loop(self.syrk_base, 'k', self.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        syrk = autofission(syrk, syrk.find('for ko in _:_ #0').after(), n_lifts=2)
        syrk = reorder_loops(syrk, 'j ko')
        syrk = reorder_loops(syrk, 'i ko')
        syrk = replace(syrk, 'for i in _:_ #0', self.gepp_syrk_base)
        syrk = call_eqv(syrk, 'gepp_syrk_base(_)', self.gepp_syrk_scheduled)
        return syrk


    def write_to_file(self, name="syrk.c"):
        ### Write syrk to a file
        file = open(f"../c/{name}", 'w+')
        file.write(self.syrk_scheduled.c_code_str())
        file.close()


microkernel = Microkernel(NeonMachine, 4, 16, 64)
gebp = GEBP_kernel(microkernel, 64)
syrk = SYRK(gebp)
syrk.write_to_file()