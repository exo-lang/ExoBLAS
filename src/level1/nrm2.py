from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *

### EXO_LOC ALGORITHM START ###
@proc
def nrm2(n: size, x: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * x[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(nrm2, "exo_" + prefix + "nrm2")
    for arg in ["x", "result"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy


def schedule_nrm2_stride_1(VEC_W, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")

    simple_stride_1 = divide_loop(
        simple_stride_1, "for i in _:_", VEC_W, ("io", "ii"), tail="cut"
    )

    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(
        stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True)
    )
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _")
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("for io in _:_").before()
    )
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("for io in _:_").after()
    )
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")

    simple_stride_1 = stage_mem(
        simple_stride_1,
        "for ii in _:_ #1",
        f"x[{VEC_W} * io: {VEC_W} * (io + 1)]",
        "xReg",
    )
    simple_stride_1 = bind_expr(simple_stride_1, "xReg[_] #1", "xReg1")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _", n_lifts=2)
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("xReg1[_] = _").after()
    )
    simple_stride_1 = simplify(simple_stride_1)

    for buffer in ["xReg", "xReg1", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, "f32")

    simple_stride_1 = replace_all_stmts(simple_stride_1, instructions)

    return simplify(simple_stride_1)


def schedule_nrm2_stride_1_interleaved(
    VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")

    simple_stride_1 = divide_loop(
        simple_stride_1,
        "for i in _:_",
        VEC_W * INTERLEAVE_FACTOR,
        ("io", "ii"),
        tail="cut",
    )
    simple_stride_1 = divide_loop(
        simple_stride_1, "for ii in _:_", VEC_W, ("im", "ii"), perfect=True
    )

    simple_stride_1 = reorder_loops(simple_stride_1, "io im")
    simple_stride_1 = reorder_loops(simple_stride_1, "io ii")
    simple_stride_1 = simplify(
        stage_mem(simple_stride_1, "for io in _:_", "result", "resultReg", accum=True)
    )
    simple_stride_1 = expand_dim(simple_stride_1, "resultReg :_ ", VEC_W, "ii")
    simple_stride_1 = expand_dim(
        simple_stride_1, "resultReg :_ ", INTERLEAVE_FACTOR, "im"
    )
    simple_stride_1 = lift_alloc(simple_stride_1, "resultReg : _", n_lifts=2)
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("for io in _:_").before(), n_lifts=2
    )
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("for io in _:_").after(), n_lifts=2
    )
    simple_stride_1 = reorder_loops(simple_stride_1, "ii io")
    simple_stride_1 = reorder_loops(simple_stride_1, "im io")

    lower_bound = f"{VEC_W} * im + {VEC_W * INTERLEAVE_FACTOR} * io"
    simple_stride_1 = stage_mem(
        simple_stride_1,
        "for ii in _:_ #1",
        f"x[{lower_bound}: {lower_bound} + {VEC_W}]",
        "xReg",
    )
    simple_stride_1 = bind_expr(simple_stride_1, "xReg[_] #1", "xReg1")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _", n_lifts=1)
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("xReg1[_] = _").after()
    )
    simple_stride_1 = simplify(simple_stride_1)

    for buffer in ["xReg", "xReg1", "resultReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)

    simple_stride_1 = replace_all_stmts(simple_stride_1, instructions)

    simple_stride_1 = expand_dim(simple_stride_1, "xReg", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg : _")
    simple_stride_1 = expand_dim(simple_stride_1, "xReg1", INTERLEAVE_FACTOR, "im")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg1 : _")

    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for im in _:_")
        return proc

    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")
    simple_stride_1 = interleave_instructions(simple_stride_1, "im")

    return simplify(simple_stride_1)


#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_snrm2_stride_any = specialize_precision("f32")
exo_snrm2_stride_any = rename(
    exo_snrm2_stride_any, exo_snrm2_stride_any.name() + "_stride_any"
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.set_zero_instr_f32,
    C.Machine.fmadd_reduce_instr_f32,
    C.Machine.reg_copy_instr_f32,
    C.Machine.assoc_reduce_add_instr_f32,
]

if None not in f32_instructions:
    exo_snrm2_stride_1 = schedule_nrm2_stride_1_interleaved(
        C.Machine.f32_vec_width,
        C.Machine.vec_units * 2,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
else:
    exo_snrm2_stride_1 = specialize_precision("f32")
    exo_snrm2_stride_1 = rename(
        exo_snrm2_stride_1, exo_snrm2_stride_1.name() + "_stride_1"
    )

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dnrm2_stride_any = specialize_precision("f64")
exo_dnrm2_stride_any = rename(
    exo_dnrm2_stride_any, exo_dnrm2_stride_any.name() + "_stride_any"
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.set_zero_instr_f64,
    C.Machine.fmadd_reduce_instr_f64,
    C.Machine.reg_copy_instr_f64,
    C.Machine.assoc_reduce_add_instr_f64,
]

if None not in f64_instructions:
    exo_dnrm2_stride_1 = schedule_nrm2_stride_1_interleaved(
        C.Machine.f32_vec_width // 2,
        C.Machine.vec_units * 2,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
else:
    exo_dnrm2_stride_1 = specialize_precision("f64")
    exo_dnrm2_stride_1 = rename(
        exo_dnrm2_stride_1, exo_dnrm2_stride_1.name() + "_stride_1"
    )
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_snrm2_stride_any,
    exo_snrm2_stride_1,
    exo_dnrm2_stride_any,
    exo_dnrm2_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
