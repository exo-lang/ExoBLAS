from __future__ import annotations

from email.mime import base
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *
from composed_schedules import interleave_execution


class Microkernel:

    microkernel_id = 0

    def __init__(
        self,
        machine: MachineParameters,
        M_r: int,
        N_r: int,
        K_blk: int,
        precision: str = "f32",
    ):

        # Problem dimensions and precision
        self.M_r = M_r
        self.N_r = N_r
        self.K_blk = K_blk
        self.precision = precision
        self.this_id = Microkernel.microkernel_id

        # Base SGEMM procedure
        @proc
        def SGEMM(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            assert stride(C, 1) == 1
            assert stride(A, 1) == 1
            assert stride(B, 1) == 1
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * B[k, j]

        @proc
        def SGEMM2(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[M, K] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            assert N < 32
            assert stride(C, 1) == 1
            assert stride(A, 1) == 1
            assert stride(B, 1) == 1
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[i, k] * B[k, j]

        @proc
        def SGEMM_TRANSPOSEA(
            M: size,
            N: size,
            K: size,
            C: f32[M, N] @ DRAM,
            A: f32[K, M] @ DRAM,
            B: f32[K, N] @ DRAM,
        ):
            assert K == M
            assert N == M
            assert stride(C, 1) == 1
            assert stride(A, 1) == 1
            assert stride(B, 1) == 1
            for i in seq(0, M):
                for j in seq(0, N):
                    for k in seq(0, K):
                        C[i, j] += A[k, i] * B[k, j]

        self.sgemm_window = self.generate_sgemm_window(SGEMM, "sgemm_win")
        self.sgemm_window2 = self.generate_sgemm_window(SGEMM2, "sgemm_win2")
        self.sgemm_window_transa = self.generate_sgemm_window(
            SGEMM_TRANSPOSEA, "sgemm_win_transa"
        )

        self.sgemm_base = SGEMM
        self.scheduled_microkernel, self.base_microkernel = self.generate_microkernel(
            machine, M_r, N_r, K_blk
        )
        (
            self.scheduled_zpad_microkernel,
            self.base_zpad_microkernel,
        ) = self.generate_microkernel_zpad(
            machine, 8 if precision == "f32" else 16, N_r, K_blk
        )

    def generate_sgemm_window(self, proc, name):
        proc = rename(proc, name)
        proc = set_window(proc, "A", True)
        proc = set_window(proc, "B", True)
        proc = set_window(proc, "C", True)

        return proc

    def generate_microkernel(
        self, machine: MachineParameters, M_r: int, N_r: int, K_blk: int
    ):
        """
        Generate microkernel for MxN problem size for the given machine.
        Vectorizes along dimension N or dimension K.
        """

        scheduled_microkernel, microkernel = self.do_generate_microkernel(
            machine, M_r, N_r, K_blk
        )

        Microkernel.microkernel_id += 1  # Ensure each microkernel has a unique name

        return scheduled_microkernel, microkernel

    def generate_base_microkernel(self, M_r: int, N_r: int, K_blk: int):

        microkernel = rename(self.sgemm_window, f"microkernel_{self.microkernel_id}")
        microkernel = microkernel.partial_eval(M=M_r, N=N_r, K=K_blk)
        return microkernel

    def do_generate_microkernel(
        self, machine: MachineParameters, M_r: int, N_r: int, K_blk: int
    ):
        """
        Vectorize along dimension N
        """
        microkernel = self.generate_base_microkernel(M_r, N_r, K_blk)
        microkernel = self.specialize_microkernel(microkernel, self.precision)

        # Reorder and divide loops
        scheduled_microkernel = rename(
            microkernel, f"{machine.name}_microkernel_{M_r}x{N_r}_{self.microkernel_id}"
        )
        scheduled_microkernel = reorder_loops(scheduled_microkernel, "j k")
        scheduled_microkernel = reorder_loops(scheduled_microkernel, "i k")
        scheduled_microkernel = divide_loop(
            scheduled_microkernel, "j", machine.vec_width, ["jo", "ji"], perfect=True
        )

        # Create C buffer in vector mem
        c_reg_str = f"C[i, {machine.vec_width} * jo + ji]"
        scheduled_microkernel = stage_mem(
            scheduled_microkernel, "C[_] += _", c_reg_str, "C_reg"
        )
        scheduled_microkernel = set_memory(
            scheduled_microkernel, "C_reg", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "C_reg",
            machine.vec_width,
            "ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "C_reg",
            N_r // machine.vec_width,
            f"jo",
            unsafe_disable_checks=True,
        )

        scheduled_microkernel = expand_dim(
            scheduled_microkernel, "C_reg", M_r, "i", unsafe_disable_checks=True
        )
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "C_reg", n_lifts=4)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("C_reg[_] = _").after(),
            n_lifts=4,
        )
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("C[_] = _").before(),
            n_lifts=4,
        )

        # Setup A buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, "A[_]", "A_vec")
        scheduled_microkernel = set_memory(
            scheduled_microkernel, "A_vec", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "A_vec",
            machine.vec_width,
            "ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel, "A_vec", M_r, "i", unsafe_disable_checks=True
        )
        scheduled_microkernel = set_precision(
            scheduled_microkernel, "A_vec", self.precision
        )

        # Setup B buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, "B[_]", "B_vec")

        scheduled_microkernel = set_memory(
            scheduled_microkernel, "B_vec", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "B_vec",
            machine.vec_width,
            f"ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "B_vec",
            (N_r // machine.vec_width),
            f"jo",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = set_precision(
            scheduled_microkernel, "B_vec", self.precision
        )

        # Move A_vec and B_vec into proper sites
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "A_vec", n_lifts=4)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("A_vec[_] = _").after(),
            n_lifts=3,
        )
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "B_vec", n_lifts=4)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("B_vec[_] = _").after(),
            n_lifts=3,
        )


        #Unroll loops
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "jo #0") #C load
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "i  #0") #C load

        scheduled_microkernel = unroll_loop(scheduled_microkernel, "i #0") #A load
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "jo #0") #B load
        
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "jo #0") #computation 
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "i  #0") #computation
        
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "jo #0") #C store
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "i  #0") #C store
        
        # Replace
        if self.precision == "f32":
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.load_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.broadcast_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.store_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.fmadd_instr_f32
            )
            scheduled_microkernel = simplify(scheduled_microkernel)
        else:
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.load_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.broadcast_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.store_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.fmadd_instr_f64
            )
            scheduled_microkernel = simplify(scheduled_microkernel)


        k_loop = scheduled_microkernel.find_loop('k')
        scheduled_microkernel = divide_loop(scheduled_microkernel, k_loop, 4, ["ko", "ki"], tail="cut", perfect=True)
        first = scheduled_microkernel.find_loop('ki').body()[0]
        for i in range(len( scheduled_microkernel.find_loop('ki').body())):
            while True:
                try:
                    scheduled_microkernel = reorder_stmts(scheduled_microkernel, scheduled_microkernel.forward(first).expand(0, 1))
                except:
                    break
            first = scheduled_microkernel.find_loop('ki').body()[0]
#
        scheduled_microkernel = unroll_loop(scheduled_microkernel, "ki")
#        print(scheduled_microkernel)
#        scheduled_microkernel = interleave_execution(scheduled_microkernel, k_loop, 4)
#        scheduled_microkernel = simplify(scheduled_microkernel)
#        print(scheduled_microkernel)

        # Unsafe assert
#        scheduled_microkernel = scheduled_microkernel.add_assertion(f"stride(A, 0) == {K_blk}")
#        scheduled_microkernel = scheduled_microkernel.add_assertion(f"stride(B, 0) == {N_r}")
#        scheduled_microkernel = scheduled_microkernel.add_assertion(f"stride(C, 0) == {N_r}")
#        microkernel = microkernel.add_assertion(f"stride(A, 0) == {K_blk}")
#        microkernel = microkernel.add_assertion(f"stride(B, 0) == {N_r}")
#        microkernel = microkernel.add_assertion(f"stride(C, 0) == {N_r}")
#        scheduled_microkernel.unsafe_assert_eq(microkernel)
        #
        
        return scheduled_microkernel, microkernel

    def generate_microkernel_zpad(
        self,
        machine: MachineParameters,
        M_r: int,
        N_r: int,
        K_blk: int,
        M_blk: int = 32,
    ):
        base_microkernel = rename(
            self.sgemm_window2, f"microkernelzpad_{self.microkernel_id}"
        )
        base_microkernel = base_microkernel.partial_eval(M=M_r, K=K_blk)
        base_microkernel = self.specialize_microkernel(base_microkernel, self.precision)

        # Reorder and divide loops
        scheduled_microkernel = rename(
            base_microkernel,
            f"{machine.name}_microkernelzpad_{M_r}x{N_r}_{self.microkernel_id}",
        )
        scheduled_microkernel = reorder_loops(scheduled_microkernel, "j k")
        scheduled_microkernel = reorder_loops(scheduled_microkernel, "i k")
        scheduled_microkernel = divide_loop(
            scheduled_microkernel, "j", machine.vec_width, ["jo", "ji"], tail="cut"
        )

        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("for ji in _:_ #1").before(),
            n_lifts=2,
        )

        # Create C buffer in vector mem
        c_reg_str = f"C[i, {machine.vec_width} * jo + ji]"
        scheduled_microkernel = stage_mem(
            scheduled_microkernel, "C[_] += _ #0", c_reg_str, "C_reg"
        )
        scheduled_microkernel = set_memory(
            scheduled_microkernel, "C_reg", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "C_reg",
            machine.vec_width,
            "ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "C_reg",
            M_blk // machine.vec_width,
            f"jo",
            unsafe_disable_checks=True,
        )

        scheduled_microkernel = expand_dim(
            scheduled_microkernel, "C_reg", M_r, "i", unsafe_disable_checks=True
        )
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "C_reg", n_lifts=4)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("C_reg[_] = _").after(),
            n_lifts=4,
        )
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("C[_] = _").before(),
            n_lifts=4,
        )

        # Setup A buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, "A[_]", "A_vec")
        scheduled_microkernel = set_memory(
            scheduled_microkernel, "A_vec", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "A_vec",
            machine.vec_width,
            "ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel, "A_vec", M_r, "i", unsafe_disable_checks=True
        )
        scheduled_microkernel = set_precision(
            scheduled_microkernel, "A_vec", self.precision
        )

        # Setup B buffer in vector mem
        scheduled_microkernel = bind_expr(scheduled_microkernel, "B[_]", "B_vec")

        scheduled_microkernel = set_memory(
            scheduled_microkernel, "B_vec", machine.mem_type
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "B_vec",
            machine.vec_width,
            f"ji",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = expand_dim(
            scheduled_microkernel,
            "B_vec",
            (M_blk // machine.vec_width),
            f"jo",
            unsafe_disable_checks=True,
        )
        scheduled_microkernel = set_precision(
            scheduled_microkernel, "B_vec", self.precision
        )

        # Move A_vec and B_vec into proper sites
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "A_vec", n_lifts=3)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("A_vec[_] = _").after(),
            n_lifts=3,
        )
        scheduled_microkernel = lift_alloc(scheduled_microkernel, "B_vec", n_lifts=3)
        scheduled_microkernel = autofission(
            scheduled_microkernel,
            scheduled_microkernel.find("B_vec[_] = _").after(),
            n_lifts=3,
        )

        # Replace
        if self.precision == "f32":
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.load_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.broadcast_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.store_instr_f32
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.fmadd_instr_f32
            )
            scheduled_microkernel = simplify(scheduled_microkernel)
        else:
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.load_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.broadcast_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.store_instr_f64
            )
            scheduled_microkernel = replace_all(
                scheduled_microkernel, machine.fmadd_instr_f64
            )
            scheduled_microkernel = simplify(scheduled_microkernel)

        return scheduled_microkernel, base_microkernel

    def specialize_microkernel(self, microkernel: Procedure, precision: str):
        args = ["A", "B", "C"]
        for arg in args:
            microkernel = set_precision(microkernel, arg, precision)
        return microkernel


# test_microkernel = Microkernel(NeonMachine, 4, 16, 32)
# print("TEST PASSED: Microkernel successfully generated!")


"""
TODO: Generate transpose version of all of these
"""


class GEBP_kernel:

    gebp_id = 0

    def __init__(
        self, microkernel: Microkernel, M_blk: int, N_blk: int, precision: str = "f32"
    ):

        # Problem dimensions
        self.M_blk = M_blk
        self.N_blk = N_blk
        self.microkernel = microkernel
        self.precision = precision
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
                        C[i, j] += A[i, k] * B[k, j]

        self.sgemm_window = rename(SGEMM, "sgemm_win")
        self.sgemm_window = set_window(self.sgemm_window, "A", True)
        self.sgemm_window = set_window(self.sgemm_window, "B", True)
        self.sgemm_window = set_window(self.sgemm_window, "C", True)

        self.sgemm_base = SGEMM

        self.scheduled_gebp, self.base_gebp = self.generate_gebp()

    def generate_base_gebp(self, M_blk: int, K_blk: int, N_blk: int):
        base_microkernel = rename(self.sgemm_window, f"gebp_base_{self.this_id}")
        return simplify(base_microkernel.partial_eval(M=M_blk, K=K_blk, N=N_blk))

    def generate_gebp(self):

        scheduled_gebp, gebp = self.do_generate_gebp()

        GEBP_kernel.gebp_id += 1

        return scheduled_gebp, gebp

    def do_generate_gebp(self):

        gebp = self.generate_base_gebp(self.M_blk, self.microkernel.K_blk, self.N_blk)
        gebp = self.specialize_gebp(gebp, self.precision)

        scheduled_gebp = rename(
            gebp, f"gebp_{self.M_blk}x{self.microkernel.K_blk}_{self.gebp_id}"
        )
        scheduled_gebp = divide_loop(
            scheduled_gebp,
            "i",
            self.microkernel.M_r,
            ["io", "ii"],
            tail="cut_and_guard",
        )
        scheduled_gebp = simplify(
            divide_loop(
                scheduled_gebp, "j", self.microkernel.N_r, ["jo", "ji"], perfect=True
            )
        )

        # scheduled_gebp = autofission(scheduled_gebp, scheduled_gebp.find('for jo in _: _').after(), n_lifts=2)
        scheduled_gebp = reorder_loops(scheduled_gebp, "ii jo")
        scheduled_gebp = reorder_loops(scheduled_gebp, "io jo")
        #scheduled_gebp = divide_loop(scheduled_gebp, "io", 2, ["ioo", "ioi"], perfect=True)

        #print(scheduled_gebp)
        scheduled_gebp = stage_mem(
            scheduled_gebp,
            "for io in _:_ #0",
            f"B[0:{self.microkernel.K_blk}, jo*{self.microkernel.N_r}:jo*{self.microkernel.N_r} + {self.microkernel.N_r}]",
            "B_reg_strip",
        )
        scheduled_gebp = simplify(scheduled_gebp)
        #print(scheduled_gebp)
        #scheduled_gebp = unroll_loop(scheduled_gebp, "i1")

        scheduled_gebp = stage_mem(
            scheduled_gebp,
            "for jo in _:_ #0",
            f"A[0:{self.M_blk}, 0:{self.microkernel.K_blk}]",
            "A_strip",
        )
        scheduled_gebp = simplify(scheduled_gebp)
        scheduled_gebp = divide_dim(scheduled_gebp, "A_strip:_", 0, self.microkernel.M_r)
        scheduled_gebp = rearrange_dim(scheduled_gebp, "A_strip:_", [0, 2, 1])
        scheduled_gebp = simplify(scheduled_gebp)

        scheduled_gebp = replace(scheduled_gebp, "for ii in _:_", self.microkernel.base_microkernel)
        call_c = scheduled_gebp.find(f"microkernel_{self.microkernel.this_id}(_)")
        scheduled_gebp = call_eqv(
            scheduled_gebp,
            call_c,
            self.microkernel.scheduled_microkernel,
        )


        scheduled_gebp = inline(scheduled_gebp, call_c)
        scheduled_gebp = inline_window(scheduled_gebp, "C = C[_]")
        scheduled_gebp = inline_window(scheduled_gebp, "A = A_strip[_]")
        #scheduled_gebp = inline_window(scheduled_gebp, "A = A[_]")
        scheduled_gebp = inline_window(scheduled_gebp, "B = B_reg_strip[_]")
        scheduled_gebp = simplify(scheduled_gebp)

        # Unsafe assert
#        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(A, 0) == {self.microkernel.K_blk}")
#        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(B, 0) == {self.N_blk}")
#        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(C, 0) == {self.N_blk}")
#        gebp = gebp.add_assertion(f"stride(A, 0) == {self.microkernel.K_blk}")
#        gebp = gebp.add_assertion(f"stride(B, 0) == {self.N_blk}")
#        gebp = gebp.add_assertion(f"stride(C, 0) == {self.N_blk}")
        #

        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(A, 1) == 1")
        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(B, 1) == 1")
        scheduled_gebp = scheduled_gebp.add_assertion(f"stride(C, 1) == 1")
        gebp = gebp.add_assertion(f"stride(A, 1) == 1")
        gebp = gebp.add_assertion(f"stride(B, 1) == 1")
        gebp = gebp.add_assertion(f"stride(C, 1) == 1")
        scheduled_gebp.unsafe_assert_eq(gebp)

        return scheduled_gebp, gebp

    def specialize_gebp(self, gebp: Procedure, precision: str):
        args = ["A", "B", "C"]
        for arg in args:
            gebp = set_precision(gebp, arg, precision)
        return gebp


class GEPP_kernel:

    gepp_id = 0

    def __init__(self, gebp: GEBP_kernel, precision: str = "f32"):

        self.K_blk = gebp.microkernel.K_blk
        self.N_blk = gebp.N_blk
        self.precision = precision
        self.gebp = gebp
        self.this_id = GEPP_kernel.gepp_id

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
                        C[i, j] += A[i, k] * B[k, j]

        self.sgemm_window = rename(SGEMM, "sgemm_win")
        self.sgemm_window = set_window(self.sgemm_window, "A", True)
        self.sgemm_window = set_window(self.sgemm_window, "B", True)
        self.sgemm_window = set_window(self.sgemm_window, "C", True)

        self.sgemm_base = SGEMM

        self.gepp_base, self.gepp_scheduled = self.generate_gepp()

    def generate_base_gepp(self):
        base_gepp = rename(self.sgemm_window, f"gepp_base_{self.this_id}")
        return base_gepp.partial_eval(K=self.K_blk, N=self.N_blk)

    def generate_gepp(self):
        base_gepp, scheduled_gepp = self.do_generate_gepp()

        GEPP_kernel.gepp_id += 1

        return base_gepp, scheduled_gepp

    def do_generate_gepp(self):

        base_gepp = self.generate_base_gepp()
        base_gepp = self.specialize_gepp(base_gepp, self.precision)

        scheduled_gepp = rename(base_gepp, f"scheduled_gepp_{self.this_id}")
        scheduled_gepp = divide_loop(
            scheduled_gepp, "i", self.gebp.M_blk, ["io", "ii"], tail="cut_and_guard"
        )
        # scheduled_gepp = stage_mem(scheduled_gepp, 'for ii in _:_ #0', f'B[0:{self.K_blk}, 0:N]', 'B_packed')

        scheduled_gepp = replace(
            scheduled_gepp, "for ii in _:_ #0", self.gebp.base_gebp
        )
        call_c = scheduled_gepp.find(f"gebp_base_{self.gebp.this_id}(_)")
        scheduled_gepp = call_eqv(
            scheduled_gepp,
            call_c,
            self.gebp.scheduled_gebp,
        )
        scheduled_gepp = inline(scheduled_gepp, call_c)
        scheduled_gepp = inline_window(scheduled_gepp, "C = C[_]")
        scheduled_gepp = inline_window(scheduled_gepp, "A = A[_]")
        scheduled_gepp = inline_window(scheduled_gepp, "B = B[_]")


        # Unsafe assert
#        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(A, 0) == {self.K_blk}")
#        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(B, 0) == {self.N_blk}")
#        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(C, 0) == {self.N_blk}")
#        base_gepp = base_gepp.add_assertion(f"stride(A, 0) == {self.K_blk}")
#        base_gepp = base_gepp.add_assertion(f"stride(B, 0) == {self.N_blk}")
#        base_gepp = base_gepp.add_assertion(f"stride(C, 0) == {self.N_blk}")
        # 

        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(A, 1) == 1")
        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(B, 1) == 1")
        scheduled_gepp = scheduled_gepp.add_assertion(f"stride(C, 1) == 1")
        base_gepp = base_gepp.add_assertion(f"stride(A, 1) == 1")
        base_gepp = base_gepp.add_assertion(f"stride(B, 1) == 1")
        base_gepp = base_gepp.add_assertion(f"stride(C, 1) == 1")
        scheduled_gepp.unsafe_assert_eq(base_gepp)

        return base_gepp, scheduled_gepp

    def specialize_gepp(self, gepp: Procedure, precision: str):
        args = ["A", "B", "C"]
        for arg in args:
            gepp = set_precision(gepp, arg, precision)
        return gepp


# test_gebp = GEBP_kernel(test_microkernel, 64, 256)
