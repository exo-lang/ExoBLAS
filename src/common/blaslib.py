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
from codegen_helpers import *


def optimize_level_1(proc, loop, precision, machine, interleave_factor, vec_tail=None):
    vec_width = machine.vec_width(precision)
    memory = machine.mem_type
    instructions = machine.get_instructions(precision)

    if vec_tail is None:
        vectorize_tail = memory in {AVX2}
        vec_tail = "cut_and_predicate" if vectorize_tail else "cut"

    loop = proc.forward(loop)
    proc = cse(proc, loop.body(), precision)

    rules = [fma_rule, abs_rule]
    proc, (loop,) = vectorize(
        proc, loop, vec_width, precision, memory, rules=rules, tail=vec_tail, rc=True
    )

    proc, (_, loop) = hoist_from_loop(proc, loop, rc=True)

    proc = interleave_loop(
        proc, loop, interleave_factor, par_reduce=True, memory=memory
    )

    proc = cleanup(proc)
    proc = replace_all_stmts(proc, instructions)
    return proc


def get_triangle_type(proc, loop):
    loop = proc.forward(loop)
    outer_loop = loop.parent()
    assert is_loop(proc, outer_loop)

    if not is_literal(proc, loop.lo(), 0):
        return 0
    if is_read(proc, loop.hi(), outer_loop.name()):
        return 1
    if not is_add(proc, loop.hi()):
        return 0
    if is_read(proc, loop.hi().lhs(), outer_loop.name()) and is_literal(
        proc, loop.hi().rhs(), 1
    ):
        return 2
    return 0


def optimize_level_2(
    proc, outer_loop, precision, machine, rows_factor, cols_factor, round_up=None
):
    vec_width = machine.vec_width(precision)
    memory = machine.mem_type
    inner_loop = get_inner_loop(proc, outer_loop)

    if triangle := get_triangle_type(proc, inner_loop):
        if round_up is None:
            round_up = memory in {AVX2}
        rows_factor = min(rows_factor, vec_width)
        if round_up and triangle == 1:
            proc, (outer_loop,) = cut_loop_and_unroll(proc, outer_loop, 1, rc=True)
            inner_loop = get_inner_loop(proc, outer_loop)
        if not round_up and triangle == 2:
            proc, (inner_loop,) = cut_loop_and_unroll(
                proc, inner_loop, 1, front=False, rc=True
            )
        proc = round_loop(proc, inner_loop, vec_width, up=round_up)

    proc = parallelize_all_reductions(proc, inner_loop, 1, unroll=True)
    proc, (outer_loop_o, _, _) = auto_divide_loop(
        proc, outer_loop, rows_factor, tail="cut"
    )
    proc = simplify(proc)
    proc = unroll_and_jam_parent(proc, inner_loop, rows_factor)
    proc = unroll_buffers(proc, outer_loop_o)
    proc = optimize_level_1(proc, inner_loop, precision, machine, cols_factor)
    return proc


__all__ = ["optimize_level_1", "optimize_level_2"]
