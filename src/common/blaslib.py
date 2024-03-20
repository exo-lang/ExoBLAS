from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from stdlib import *
from inspection import *
from codegen_helpers import *


def optimize_level_1(
    proc,
    loop,
    precision,
    machine,
    interleave_factor,
    instrs=None,
    vec_tail=None,
    inter_tail="recursive",
):
    vec_width = machine.vec_width(precision)
    memory = machine.mem_type
    if instrs is None:
        instrs = machine.get_instructions(precision)

    if vec_tail is None:
        vec_tail = "cut_and_predicate" if machine.supports_predication else "cut"

    loop = proc.forward(loop)
    proc = cse(proc, loop.body(), precision)

    rules = [fma_rule, abs_rule]
    proc, (loop,) = vectorize(
        proc, loop, vec_width, precision, memory, rules=rules, tail=vec_tail, rc=True
    )

    proc, (_, loop) = hoist_from_loop(proc, loop, rc=True)

    proc = interleave_loop(
        proc, loop, interleave_factor, par_reduce=True, memory=memory, tail=inter_tail
    )

    proc = cleanup(proc)
    proc = replace_all_stmts(proc, instrs)
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
    proc,
    outer_loop,
    precision,
    machine,
    rows_factor,
    cols_factor,
    round_up=None,
    rows_tail="cut",
    **kwargs,
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

    def rewrite(proc, outer_loop, rows_factor, cols_factor):
        kernel_loop = outer_loop.parent()
        inner_loop = get_inner_loop(proc, outer_loop)
        proc = parallelize_all_reductions(proc, inner_loop, 1, unroll=True)
        proc = unroll_and_jam_parent(proc, inner_loop, rows_factor)
        proc = unroll_buffers(proc, kernel_loop)
        proc = optimize_level_1(
            proc, inner_loop, precision, machine, cols_factor, **kwargs
        )
        return proc

    if rows_tail in {"level_1", "cut"}:
        proc, (_, inner, tail_l) = divide_loop_(
            proc, outer_loop, rows_factor, tail="cut", rc=True
        )
        proc = rewrite(proc, inner, rows_factor, cols_factor)
        if rows_tail == "level_1":
            proc = optimize_level_1(
                proc,
                get_inner_loop(proc, proc.body()[-1]),
                precision,
                machine,
                rows_factor,
            )
    return proc


__all__ = ["optimize_level_1", "optimize_level_2"]
