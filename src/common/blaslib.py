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


def optimize_level_1(proc, loop, params):
    vec_width = params.vec_width
    mem_type = params.mem_type
    precision = params.precision
    instructions = params.instructions
    interleave_factor = params.interleave_factor

    loop = proc.forward(loop)

    # Determine the tail strategy
    vectorize_tail = mem_type in {AVX2}
    tail = "guard" if vectorize_tail else "cut"

    # Tile to exploit vectorization
    proc, (outer_loop, inner_loop, _) = auto_divide_loop(
        proc, loop, vec_width, tail=tail
    )

    # Parallelize all reductions
    for stmt in lrn_stmts(proc, inner_loop.body()):
        try:
            proc = parallelize_reduction(proc, stmt, mem_type)
        except (SchedulingError, BLAS_SchedulingError, TypeError):
            pass

    # Previous step calls fission which would change what
    # inner loop we are pointing at
    outer_loop = proc.forward(outer_loop)
    inner_loop = outer_loop.body()[0]

    # Generate simd
    proc = scalar_to_simd(proc, inner_loop, vec_width, mem_type, precision)

    # Generate the vectorized tail
    if vectorize_tail:
        # Cut the tail iterations
        last_outer_iteration = FormattedExprStr("_ - 1", outer_loop.hi())
        proc = cut_loop(proc, outer_loop, last_outer_iteration)

        # Eliminate the predication in the main inner loop
        proc = dce(proc, outer_loop)

    if interleave_factor == 1:
        return simplify(replace_all(proc, instructions))

    # Tile to exploit ILP
    proc, (outer_loop, inner_loop, _) = auto_divide_loop(
        proc, outer_loop, interleave_factor, tail="cut"
    )

    # Parallelize all reductions
    for stmt in lrn_stmts(proc, inner_loop.body()):
        try:
            proc = parallelize_reduction(proc, stmt, mem_type, 3, True)
        except (SchedulingError, BLAS_SchedulingError, TypeError):
            pass

    # Intereleave to increase ILP
    inner_loop = proc.forward(outer_loop).body()[0]
    proc = interleave_execution(proc, inner_loop, interleave_factor)

    # Instructions Selection
    proc = replace_all(proc, instructions)

    proc = simplify(proc)
    return proc


__all__ = [
    "optimize_level_1",
]