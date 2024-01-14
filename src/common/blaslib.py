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

    proc = parallelize_all_reductions(proc, inner_loop, mem_type)

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

    # Hoist any stmt
    index_from_end = get_index_in_body(proc, outer_loop, False)
    proc = hoist_from_loop(proc, outer_loop)
    outer_loop = get_parent(proc, outer_loop).body()[index_from_end]

    if interleave_factor == 1:
        return simplify(replace_all(proc, instructions))

    # Tile to exploit ILP
    proc, (outer_loop, inner_loop, _) = auto_divide_loop(
        proc, outer_loop, interleave_factor, tail="cut"
    )

    proc = parallelize_all_reductions(proc, inner_loop, mem_type, unroll=True)

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
