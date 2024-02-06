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

    proc = cse(proc, loop.body())

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


def optimize_level_2(proc, outer_loop, params):
    rows_factor = params.rows_interleave_factor
    vec_width = params.vec_width
    inner_loop = get_inner_loop(proc, outer_loop)

    is_trianglular = is_literal(proc, inner_loop.lo(), 0) and is_read(
        proc, inner_loop.hi(), outer_loop.name()
    )
    if is_trianglular:
        rows_factor = min(rows_factor, vec_width)
        if params.mem_type in {AVX2}:
            proc, (const_loop, outer_loop) = cut_loop_(proc, outer_loop, 1, rc=True)
            inner_loop = get_inner_loop(proc, outer_loop)
            proc = unroll_loop(proc, const_loop)
            proc = shift_loop(proc, outer_loop, 0)
            proc = round_loop(proc, inner_loop, params.vec_width, up=True)
        else:
            proc = round_loop(proc, inner_loop, params.vec_width, up=False)

    proc = parallelize_all_reductions(proc, inner_loop, 1, unroll=True)
    proc, (outer_loop_o, _, _) = auto_divide_loop(
        proc, outer_loop, rows_factor, tail="cut"
    )
    proc = simplify(proc)
    proc = unroll_and_jam_parent(proc, inner_loop, rows_factor)
    proc = unroll_buffers(proc, outer_loop_o)
    proc = optimize_level_1(proc, inner_loop, params)
    return simplify(proc)


__all__ = ["optimize_level_1", "optimize_level_2"]
