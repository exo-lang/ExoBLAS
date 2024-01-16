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
    proc = parallelize_all_reductions(proc, proc.find_loop("jo"), params.mem_type, 2)
    proc = unroll_and_jam_parent(
        proc, proc.find_loop("jo"), params.rows_interleave_factor, (True, False, True)
    )

    # Data reuse across rows
    proc = simplify(auto_stage_mem(proc, proc.find(reuse), "shared", n_lifts=2))
    proc = set_memory(proc, "shared", AVX2)  # Simply to avoid a vector copy

    # Generate SIMD
    proc = scalar_to_simd(
        proc, proc.find_loop("ii").body()[0], params.vec_width, AVX2, params.precision
    )

    # Interleave multiple rows dots
    proc = interleave_execution(
        proc, proc.find_loop("ii"), params.rows_interleave_factor
    )

    # Separate the tail case
    if vectorize_tail:
        loop = proc.find_loop("jo")
        proc = cut_loop(proc, loop, FormattedExprStr("_ - 1", loop.hi()))
        proc = dce(proc, loop)

    # Instruction Selection
    proc = replace_all(proc, params.instructions)
    return simplify(proc)


__all__ = ["optimize_level_1", "optimize_level_2"]
