from __future__ import annotations
from enum import Enum

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.frontend.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from stdlib import *
from inspection import *
from codegen_helpers import *


def blas_specialize_precision(proc, precision, all_buffs=True):
    assert precision in {"f32", "f64"}
    prefix = "s" if precision == "f32" else "d"
    template_name = proc.name()
    proc = rename(proc, prefix + template_name)
    return specialize_precision(proc, precision, all_buffs)


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
    proc, (outer_loop,) = adjust_level_2_triangular(proc, outer_loop, precision, machine, rows_factor, round_up)
    proc = unroll_and_jam(proc, outer_loop, rows_factor)
    inner_loop = get_inner_loop(proc, outer_loop)
    proc = attempt(unroll_buffers)(proc, outer_loop)
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
    rows_tail="cut",
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


def rewrite_divide_guard(proc, if_c):
    args = if_c.cond().lhs().lhs(), if_c.cond().rhs(), if_c.cond().lhs().rhs()
    proc = rewrite_expr(proc, if_c.cond(), FormattedExprStr("_ < _ - _", *args))
    return proc


def cut_level3_tail(proc, loop, fac, vw=1):
    loop = proc.forward(loop)
    if_c = loop.find("if _:_", many=True)[-1]
    cond = if_c.cond().rhs()
    cut_val = FormattedExprStr(f"_ - 1", loop.hi())
    proc, (loop1, loop2) = cut_loop_(proc, loop, cut_val, rc=True)
    op = "==" if vw == 1 else "<="
    conds = [f"{expr_to_string(cond)} {op} {vw} * {i}" for i in range(fac)]
    proc = specialize(proc, loop2.body(), conds)
    return cleanup(proc)


def decompose_into_rank_updates(proc, k_loop):
    k_loop = proc.forward(k_loop)
    should_fission = False
    while True:
        try:
            add = k_loop.find("_ += _ + _")
            proc = split_write(proc, add)
            should_fission = True
        except:
            break
    if should_fission:
        proc = fission_into_singles(proc, k_loop)
    return proc


gcache = []


def cache(func):
    def wrapper(proc, *args, **kwargs):
        global gcache
        print("Looking for: ")
        print(proc)
        for p, sched_p in gcache:
            proc_renamed = rename(proc, "dummy")
            p_renamed = rename(p, "dummy")
            if str(proc_renamed) == str(p_renamed):
                proc, success = attempt(replace)(proc, proc.body(), p, quiet=True, rs=True)
                if not success:
                    continue
                proc = call_eqv(proc, proc.body()[0], sched_p)
                print("Found in Cache")
                return proc
        print("Not found in cache")
        sched_proc = func(proc, *args, **kwargs)
        gcache.append((proc, sched_proc))
        return sched_proc

    return wrapper


@cache
def schedule_compute(proc, i_loop, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    j_loop = get_inner_loop(proc, i_loop)
    k_loop = get_inner_loop(proc, j_loop)

    writes = filter_cursors(is_write)(proc, lrn_stmts(proc, i_loop))
    writes = {w.name() for w in writes}
    assert len(writes) == 1
    output = next(iter(writes))

    proc = auto_stage_mem(proc, proc.body(), "alpha", "alpha_")

    proc = tile_loops_bottom_up(proc, i_loop, (m_r, n_r), tail="guard")
    i_loop = proc.forward(i_loop)
    proc = apply(rewrite_divide_guard)(proc, i_loop.find("if _:_", many=True))

    proc, cs = auto_stage_mem(proc, k_loop, output, "mtile", accum=True, rc=1)
    proc = set_memory(proc, cs.alloc, machine.mem_type)
    proc = repeate_n(parallelize_and_lift_alloc)(proc, cs.alloc, n=4)
    proc = divide_dim(proc, cs.alloc, 1, vw)
    proc = lift_reduce_constant(proc, cs.load.expand(0, 1))
    proc = inline_assign(proc, proc.forward(cs.store).prev())

    proc = fission(proc, proc.forward(cs.load).after(), n_lifts=4)
    proc = fission(proc, proc.forward(cs.store).before(), n_lifts=4)
    proc = repeate_n(lift_scope)(proc, k_loop, n=4)
    proc = decompose_into_rank_updates(proc, k_loop)

    loops = list(filter_cursors(is_loop)(proc, proc.forward(j_loop).body()))
    loops = [loops[0]] + [loop.body()[0] for loop in loops[1:-1]] + [loops[-1]]
    proc = apply(optimize_level_2)(proc, loops, precision, machine, m_r, n_r_fac, rows_tail="perfect", vec_tail="perfect")
    proc = inline_calls(proc)
    proc = apply(attempt(lift_scope))(proc, filter_cursors(is_if)(proc, nlr_stmts(proc)))

    proc = cut_level3_tail(proc, j_loop, n_r_fac, vw=vw)
    proc = replace_all_stmts(proc, machine.get_instructions(precision))
    proc = cut_level3_tail(proc, i_loop, m_r)

    tile = 64 // m_r // (4 if precision == "f32" else 8)
    proc = apply(divide_loop_)(proc, proc.find_loop("k", many=True), tile, tail="cut")
    return simplify(proc)


__all__ = ["optimize_level_1", "optimize_level_2", "schedule_compute", "blas_specialize_precision"]
