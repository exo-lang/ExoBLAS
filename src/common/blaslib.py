from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from composed_schedules import *
from introspection import *
from parameters import Level_1_Params
from codegen_helpers import *


def optimize_level_1(proc, loop, params):
    vec_width = params.vec_width
    mem_type = params.mem_type
    precision = params.precision
    instructions = params.instructions
    interleave_factor = params.interleave_factor

    loop = proc.forward(loop)

    # Vectorization
    vectorize_tail = mem_type in {AVX2}
    tail = "predicate" if vectorize_tail else "cut"
    proc, (loop,) = vectorize(
        proc, loop, vec_width, precision, mem_type, tail=tail, rc=True
    )

    # Hoist any stmt
    proc, (_, loop) = hoist_from_loop(proc, loop, rc=True)

    if vectorize_tail:
        proc = cut_tail_and_unguard(proc, loop)

    if interleave_factor == 1:
        return simplify(proc)

    proc = interleave_loop(
        proc, loop, interleave_factor, par_reduce=True, memory=mem_type
    )

    proc = cleanup(proc)
    proc = replace_all_stmts(proc, instructions)
    return proc


def optimize_level_2(proc, params, reuse):
    proc = generate_stride_1_proc(proc, params.precision)

    # Taking a subspace of the 2D iteration dimension
    proc, _ = auto_divide_loop(
        proc, proc.find_loop("i"), params.rows_interleave_factor, tail="cut"
    )

    # Determine the tail strategy
    vectorize_tail = params.mem_type in {AVX2}
    tail = "guard" if vectorize_tail else "cut"

    proc, _ = auto_divide_loop(proc, proc.find_loop("j"), params.vec_width, tail=tail)
    proc = parallelize_all_reductions(
        proc, proc.find_loop("jo"), memory=params.mem_type
    )
    proc = unroll_and_jam_parent(
        proc, proc.find_loop("jo"), params.rows_interleave_factor, (True, False, True)
    )

    # Data reuse across rows
    proc = simplify(auto_stage_mem(proc, proc.find(reuse), "shared", n_lifts=2))
    proc = set_memory(proc, "shared", params.mem_type)  # Simply to avoid a vector copy

    # Generate SIMD
    proc = scalar_to_simd(
        proc,
        proc.find_loop("ii").body()[0],
        params.vec_width,
        params.mem_type,
        params.precision,
    )

    # Interleave multiple rows dots
    proc = interleave_loop(proc, proc.find_loop("ii"))

    # Separate the tail case
    if vectorize_tail:
        loop = proc.find_loop("jo")
        proc = cut_loop(proc, loop, FormattedExprStr("_ - 1", loop.hi()))
        proc = dce(proc, loop)

    # Instruction Selection
    proc = replace_all_stmts(proc, params.instructions)
    return simplify(proc)


__all__ = ["optimize_level_1", "optimize_level_2"]
