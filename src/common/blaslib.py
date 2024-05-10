from __future__ import annotations
from enum import Enum

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
    patterns = machine.patterns

    if instrs is None:
        instrs = machine.get_instructions(precision)

    if vec_tail is None:
        vec_tail = "cut_and_predicate" if machine.supports_predication else "cut"

    loop = proc.forward(loop)
    proc = cse(proc, loop.body(), precision)

    proc, (loop,) = vectorize(proc, loop, vec_width, precision, memory, patterns=patterns, tail=vec_tail, rc=True)

    proc, (_, loop) = hoist_from_loop(proc, loop, rc=True)

    proc = interleave_loop(proc, loop, interleave_factor, par_reduce=True, memory=memory, tail=inter_tail)

    proc = cleanup(proc)
    proc = replace_all_stmts(proc, instrs)
    return simplify(proc)


class TRIANG_TYPE(Enum):
    DIAG = 1
    NO_DIAG = 2


def get_triangle_type(proc, loop):
    loop = proc.forward(loop)
    proc = simplify(proc)
    outer_loop = loop.parent()
    assert is_loop(proc, outer_loop)

    if not is_literal(proc, loop.lo(), 0):
        return None
    if is_read(proc, loop.hi(), outer_loop.name()):
        return TRIANG_TYPE.NO_DIAG
    if not is_add(proc, loop.hi()):
        return None

    oprs = [loop.hi().lhs(), loop.hi().rhs()]
    for opr1, opr2 in oprs, oprs[::-1]:
        if is_read(proc, opr1, outer_loop.name()) and is_literal(proc, opr2, 1):
            return TRIANG_TYPE.DIAG
    return None


def get_reused_vectors(proc, inner_loop):
    inner_loop = proc.forward(inner_loop)
    accesses = filter_cursors(is_access)(proc, lrn(proc, inner_loop))

    for acc in accesses:
        decl = get_declaration(proc, acc, acc.name())
        if hasattr(decl, "is_tensor") and decl.is_tensor() and len(decl.shape()) == 1:
            # This is a vector
            if inner_loop.name() in get_symbols(proc, acc):
                yield decl


def get_skinny_cond(proc, vec, precision, machine):
    assert len(vec.shape()) == 1
    vw = machine.vec_width(precision)
    vec_shp = expr_to_string(vec.shape()[0])
    if machine.supports_predication:
        return f"({vw - 1} + {vec_shp}) / {vw} * {vw}"
    else:
        return f"{vec_shp} / {vw} * {vw}"


def _optimize_level_2_skinny(proc, outer_loop, precision, machine, skinny_factor, cols_factor):
    inner_loop = get_inner_loop(proc, outer_loop)
    vw = machine.vec_width(precision)

    reused_vector = list(get_reused_vectors(proc, inner_loop))[0]  # We will only support one vector for now (level 2)
    proc = simplify(round_loop(proc, inner_loop, vw, up=machine.supports_predication))
    proc, (alloc, load, _, store) = auto_stage_mem(proc, outer_loop, reused_vector.name(), rc=True)
    proc = simplify(proc)
    proc = set_memory(proc, alloc, machine.mem_type)
    proc = divide_dim(proc, alloc, 0, vw)

    for stage in [load, store]:
        if not is_invalid(proc, stage):
            proc = optimize_level_1(proc, stage, precision, machine, 1)

    proc = optimize_level_1(proc, inner_loop, precision, machine, cols_factor)
    cond_lhs = get_skinny_cond(proc, reused_vector, precision, machine)
    proc = specialize(proc, proc.body(), [f"{cond_lhs} == {vw * i}" for i in range(1, skinny_factor + 1)])
    proc = unroll_loops(simplify(proc))
    return cleanup(proc)


def adjust_level_2_triangular(proc, outer_loop, precision, machine, rows_factor, round_up=None):
    outer_loop = proc.forward(outer_loop)
    inner_loop = get_inner_loop(proc, outer_loop)
    vw = machine.vec_width(precision)

    if triangle := get_triangle_type(proc, inner_loop):
        if round_up is None:
            round_up = machine.supports_predication
        if round_up and triangle == TRIANG_TYPE.NO_DIAG:
            proc, (outer_loop,) = cut_loop_and_unroll(proc, outer_loop, 1, rc=True)
            inner_loop = get_inner_loop(proc, outer_loop)
        if not round_up and triangle == TRIANG_TYPE.DIAG:
            proc, (inner_loop,) = cut_loop_and_unroll(proc, inner_loop, 1, front=False, rc=True)
        proc = simplify(round_loop(proc, outer_loop, rows_factor, up=False))
        proc = round_loop(proc, inner_loop, vw, up=round_up)
        return cleanup(proc), (outer_loop,)
    return proc, (outer_loop,)


def optimize_level_2_general(
    proc, outer_loop, precision, machine, rows_factor, cols_factor, round_up=None, rows_tail="cut", **kwargs
):
    outer_loop = proc.forward(outer_loop)
    rows_factor = min(rows_factor, machine.vec_width(precision))
    inner_loop = get_inner_loop(proc, outer_loop)
    proc, (outer_loop,) = adjust_level_2_triangular(proc, outer_loop, precision, machine, rows_factor, round_up)
    proc = unroll_and_jam(proc, outer_loop, rows_factor)
    proc = unroll_buffers(proc, proc.forward(inner_loop).parent())
    proc = optimize_level_1(proc, inner_loop, precision, machine, cols_factor, **kwargs)

    if rows_tail == "cut":
        tail = get_inner_loop(proc, proc.forward(outer_loop).next())
        proc = optimize_level_1(proc, tail, precision, machine, cols_factor)
    return cleanup(proc)


def optimize_level_2(
    proc,
    outer_loop,
    precision,
    machine,
    rows_factor,
    cols_factor,
    round_up=None,
    rows_tail="level_1",
    skinny_factor=None,
    **kwargs,
):
    vec_width = machine.vec_width(precision)
    memory = machine.mem_type

    proc = simplify(proc)
    inner_loop = get_inner_loop(proc, outer_loop)

    proc = parallelize_all_reductions(proc, inner_loop, 1, unroll=True)
    proc = unroll_buffers(proc, outer_loop)
    proc = attempt(lift_reduce_constant)(proc, proc.forward(inner_loop).expand(1, 0))

    if skinny_factor and machine.supports_predication:
        # We cannot partially stage memories for now so we need to stage the whole vector
        reused_vector = list(get_reused_vectors(proc, inner_loop))[0]  # We will only support one vector for now (level 2)
        cond_lhs = get_skinny_cond(proc, reused_vector, precision, machine)
        proc = specialize(proc, outer_loop, f"{expr_to_string(reused_vector.shape()[0])} <= {skinny_factor[0] * vec_width}")
        if_stmt = proc.forward(outer_loop.as_block())[0]

        proc = extract_and_schedule(
            _optimize_level_2_skinny,
        )(proc, if_stmt.body()[0], proc.name() + "_skinny", precision, machine, skinny_factor[0], skinny_factor[1])
        proc = extract_and_schedule(optimize_level_2_general)(
            proc,
            if_stmt.orelse()[0],
            proc.name() + "_general",
            precision,
            machine,
            rows_factor,
            cols_factor,
            round_up,
            rows_tail=rows_tail,
            **kwargs,
        )
    else:
        proc = optimize_level_2_general(
            proc, outer_loop, precision, machine, rows_factor, cols_factor, round_up=round_up, rows_tail=rows_tail, **kwargs
        )

    return proc


def schedule_compute(proc, i_loop, precision, machine, m_r, n_r_fac, small=False):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    writes = filter_cursors(is_write)(proc, lrn_stmts(proc, i_loop))
    writes = {w.name() for w in writes}
    assert len(writes) == 1
    output_buf = next(iter(writes))
    output_buf = get_declaration(proc, i_loop, output_buf)
    output_buf_dim0 = expr_to_string(output_buf.shape()[0])
    output_buf_dim1 = expr_to_string(output_buf.shape()[1])

    j_loop = get_inner_loop(proc, i_loop)
    k_loop = get_inner_loop(proc, j_loop)

    proc = auto_stage_mem(proc, proc.body(), "alpha", "alpha_")
    proc, cs = auto_stage_mem(proc, k_loop, "C", "C_tile", accum=True, rc=1)
    proc = lift_reduce_constant(proc, cs.load.expand(0, 1))
    assign = proc.forward(cs.store).prev()
    proc = inline_assign(proc, assign)
    proc = set_memory(proc, cs.alloc, machine.mem_type)

    proc = tile_loops_bottom_up(proc, i_loop, (m_r, n_r, None), tail="guard")
    proc = repeate_n(parallelize_and_lift_alloc)(proc, cs.alloc, n=4)
    if_i = proc.find("if _:_")
    proc = rewrite_expr(proc, if_i.cond(), f"ii < {output_buf_dim0} - {m_r} * io")
    if_j = proc.find("if _:_ #1")
    proc = rewrite_expr(proc, if_j.cond(), f"ji < {output_buf_dim1} - {n_r} * jo")

    proc = fission(proc, proc.forward(cs.load).after(), n_lifts=4)
    proc = fission(proc, proc.forward(cs.store).before(), n_lifts=4)
    proc = repeate_n(lift_scope)(proc, k_loop, n=4)
    proc = divide_dim(proc, cs.alloc, 1, vw)
    proc = apply(lift_scope)(proc, proc.find_loop("ji", many=True))
    proc = optimize_level_1(proc, proc.find_loop("ji"), precision, machine, n_r_fac, vec_tail="perfect")
    proc = optimize_level_1(proc, proc.find_loop("ji #1"), precision, machine, n_r_fac, vec_tail="perfect")
    proc = optimize_level_2(
        proc, proc.find_loop("ii #1"), precision, machine, m_r, n_r_fac, rows_tail="perfect", vec_tail="perfect"
    )

    def cut(proc, loop, cond, rng):
        loop = proc.forward(loop)
        cut_val = FormattedExprStr(f"_ - 1", loop.hi())
        proc, (loop1, loop2) = cut_loop_(proc, loop, cut_val, rc=True)
        proc = shift_loop(proc, loop2, 0)
        proc = rewrite_expr(proc, loop2.hi(), 1)
        proc = specialize(proc, loop2.body(), [f"{cond(loop2, i)} == {i}" for i in rng])
        proc = unroll_loop(proc, loop2)
        return proc

    right_cond = lambda l, i: f"({output_buf_dim1} - {l.name()} * {n_r} + {vw - 1}) / {vw}"
    proc = cut(proc, proc.find_loop("jo"), right_cond, range(1, n_r_fac))
    proc = dce(proc)
    proc = delete_pass(proc)
    proc = apply(attempt(lift_scope))(proc, nlr_stmts(proc))
    proc = replace_all_stmts(proc, machine.get_instructions(precision))
    bottom_cond = lambda l, i: f"-({m_r} * (({m_r - 1} + {output_buf_dim0}) / {m_r} - 1)) + {output_buf_dim0}"
    proc = cut(proc, proc.find_loop("io"), bottom_cond, range(1, m_r))
    proc = simplify(unroll_loops(proc))
    proc = dce(proc)
    proc = delete_pass(proc)

    tile = 64 // m_r // (4 if precision == "f32" else 8)
    proc = apply(divide_loop_)(proc, proc.find_loop("k", many=True), tile, tail="cut")
    return simplify(proc)


__all__ = ["optimize_level_1", "optimize_level_2", "schedule_compute"]
