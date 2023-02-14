from __future__ import annotations

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *

from kernels.gemm_kernels import GEPP_kernel, GEBP_kernel, Microkernel, NeonMachine, MachineParameters
from format_options import *

class GEMM:

    def __init__(self, machine: MachineParameters,
                 transa: ExoBlasT, transb: ExoBlasT,
                 alpha: int, beta: int,
                 K_blk: int, M_blk: int, 
                 M_reg: int, N_reg: int):

        @proc
        def sgemm_notranspose(
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

        self.microkernel = Microkernel(machine, M_reg, N_reg, K_blk)
        self.gebp = GEBP_kernel(self.microkernel, M_blk)
        self.gepp = GEPP_kernel(self.gebp)
        self.sgemm_base = sgemm_notranspose
        self.sgemm_scheduled = self.schedule_gepp_notranspose()


    def schedule_gepp_notranspose(self):
        sgemm_scheduled = divide_loop(self.sgemm_base, 'k', self.microkernel.K_blk, ['ko', 'ki'], tail='cut_and_guard')
        sgemm_scheduled = autofission(sgemm_scheduled, sgemm_scheduled.find('for ko in _:_ #0').after(), n_lifts=2)
        sgemm_scheduled = reorder_loops(sgemm_scheduled, 'j ko')
        sgemm_scheduled = reorder_loops(sgemm_scheduled, 'i ko')
        sgemm_scheduled = replace(sgemm_scheduled, 'for i in _:_ #0', self.gepp.gepp_base)
        sgemm_scheduled = call_eqv(sgemm_scheduled, 'gepp_base(_)', self.gepp.gepp_scheduled)
        return sgemm_scheduled


    def write_to_file(self, name="gemm.c"):
        ### Write syrk to a file
        file = open(f"c/gemm/{name}", 'w+')
        file.write(self.sgemm_scheduled.c_code_str())
        file.close()

if __name__=="__main__":
    # Process command line args
    args = sys.argv
    optlist, _ = getopt.getopt(args[1:], '',longopts=['kc=', 'mc=', 'mr=', 'nr='])

    k_blk = int(optlist[0][1])
    m_blk = int(optlist[1][1])
    m_reg = int(optlist[2][1])
    n_reg = int(optlist[3][1])

    sgemm = GEMM(NeonMachine, 
                ExoBlasNoTranspose, ExoBlasNoTranspose, 
                1, 1, 
                k_blk, m_blk, 
                m_reg, n_reg)

    sgemm.write_to_file()