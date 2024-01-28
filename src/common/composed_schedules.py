from __future__ import annotations
from dataclasses import dataclass

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from introspection import *
from exceptions import *
from higher_order import *


@dataclass
class fission_cursors:
    scope1: ForCursor | IfCursor
    scope2: ForCursor | IfCursor

    def __iter__(self):
        yield self.scope1
        yield self.scope2


def fission_(proc, gap, rc=False):
    gap = proc.forward(gap)
    stmt = gap.anchor()
    is_after = stmt.after() is gap
    proc = fission(proc, gap)

    scope1 = proc.forward(stmt).parent()
    scope2 = scope1.next()

    if is_after:
        scope1, scope2 = scope2, scope1

    if rc:
        return proc, fission_cursors(scope1, scope2)
    return proc


@dataclass
class bind_and_set_expr_cursors:
    alloc: AllocCursor
    bound_expr: ExprCursor
    expr_reads: ExprCursor


def bind_and_set_expr(proc, exprs, precision, memory, new_name=None, rc=False):
    if new_name is None:
        new_name = next(get_unique_names(proc))

    expr = exprs if isinstance(exprs, ExprCursor) else exprs[0]
    stmt = get_enclosing_stmt(proc, expr)
    proc = bind_expr(proc, exprs, new_name)
    proc = set_precision(proc, new_name, precision)
    proc = set_memory(proc, new_name, memory)

    alloc = get_declaration(proc, stmt, new_name)
    bound_expr = alloc.next().rhs()
    # Disabled since forwarding after replace is not supported now
    # exprs = [proc.forward(e) for e in exprs]
    return proc, bind_and_set_expr_cursors(alloc, bound_expr, exprs)


def expr_to_string(expr_cursor, subst={}):
    def expr_list_to_string(expr_list, subst):
        expr_str_list = [expr_to_string(i, subst) for i in expr_list]
        if not expr_str_list:
            return ""
        return "[" + ", ".join(expr_str_list) + "]"

    if isinstance(expr_cursor, ExprListCursor):
        return expr_list_to_string(expr_cursor, subst)

    if not isinstance(expr_cursor, ExprCursor):
        raise BLAS_SchedulingError("Cursor must be an ExprCursor")
    if isinstance(expr_cursor, ReadCursor):
        name = str(expr_cursor.name())
        if name in subst:
            return f"({subst[name]})"
        idx_str = expr_list_to_string(expr_cursor.idx(), subst)
        return f"({name}{idx_str})"
    elif isinstance(expr_cursor, ReadConfigCursor):
        raise BLAS_SchedulingError("ReadConfigCursor is not supported")
    elif isinstance(expr_cursor, LiteralCursor):
        val_str = str(expr_cursor.value())
        return f"({val_str})"
    elif isinstance(expr_cursor, UnaryMinusCursor):
        arg_str = expr_to_string(expr_cursor.arg, subst)
        return f"(-{arg_str})"
    elif isinstance(expr_cursor, BinaryOpCursor):
        binop_str = expr_cursor.op()
        lhs_str = expr_to_string(expr_cursor.lhs(), subst)
        rhs_str = expr_to_string(expr_cursor.rhs(), subst)
        return f"({lhs_str}{binop_str}{rhs_str})"
    elif isinstance(expr_cursor, BuiltInFunctionCursor):
        name = expr_cursor.name()
        args_str = expr_list_to_string(expr_cursor.args(), subst)
        return f"({name}({args_str[1:-1]}))"
    elif isinstance(expr_cursor, WindowExprCursor):
        raise BLAS_SchedulingError("WindowExprCursor is not supported")
    else:
        assert False, "Undefined Type"


def get_statement(cursor):

    while not isinstance(cursor, StmtCursor):
        cursor = cursor.parent()

    return cursor


def is_already_divided(loop_cursor, div_factor):
    return (
        len(loop_cursor.body()) == 1
        and isinstance(loop_cursor.body()[0], ForCursor)
        and isinstance(loop_cursor.body()[0].hi(), LiteralCursor)
        and loop_cursor.body()[0].hi().value() == div_factor
    )


def stage_expr(proc, expr_cursors, new_name, precision="R", memory=DRAM, n_lifts=1):
    """
    for i in seq(0, hi):
        s (e(i));

    ----->

    new_name: R[hi]
    for i in seq(0, hi):
        new_name[i] = e(i)
    for i in seq(0, hi):
        s (new_name[i]);
    """

    if not isinstance(expr_cursors, list):
        expr_cursors = [expr_cursors]

    expr_cursors = [proc.forward(c) for c in expr_cursors]
    enclosing_loop = get_enclosing_loop(proc, expr_cursors[0])
    stmt = get_statement(expr_cursors[0])
    proc = bind_expr(proc, expr_cursors, new_name)
    stmt = proc.forward(stmt)
    bind_stmt = stmt.prev()
    alloc_stmt = bind_stmt.prev()
    proc = set_precision(proc, alloc_stmt, precision)
    proc = set_memory(proc, alloc_stmt, memory)
    proc = expand_dim(
        proc, alloc_stmt, expr_to_string(enclosing_loop.hi()), enclosing_loop.name()
    )
    proc = lift_alloc(proc, alloc_stmt, n_lifts=n_lifts)
    proc = fission(proc, bind_stmt.after(), n_lifts=n_lifts)
    return proc


def parallelize_and_lift_alloc(proc, alloc_cursor, n_lifts=1):
    """
    for i in seq(0, hi):
        B1;
        name: type[shape];
        B2;

    ----->

    name: type[hi][shape]
    for i in seq(0, hi):
        B1;
        B2;
    """
    alloc_cursor = proc.forward(alloc_cursor)
    for i in range(n_lifts):
        alloc_cursor = proc.forward(alloc_cursor)
        enclosing_scope = alloc_cursor.parent()
        if isinstance(enclosing_scope, ForCursor):
            proc = expand_dim(
                proc,
                alloc_cursor,
                expr_to_string(enclosing_scope.hi()),
                enclosing_scope.name(),
            )
        proc = lift_alloc(proc, alloc_cursor)
    return proc


@dataclass
class auto_divide_loop_cursors:
    outer_loop: ForCursor
    inner_loop: ForCursor
    tail_loop: ForCursor

    def __iter__(self):
        yield self.outer_loop
        yield self.inner_loop
        yield self.tail_loop


def auto_divide_loop(proc, loop_cursor, div_const, tail="guard", perfect=False):
    loop_cursor = proc.forward(loop_cursor)
    loop_iter = loop_cursor.name()
    proc = divide_loop(
        proc,
        loop_cursor,
        div_const,
        (loop_iter + "o", loop_iter + "i"),
        tail=tail,
        perfect=perfect,
    )
    outer_loop = proc.forward(loop_cursor)
    inner_loop = outer_loop.body()[0]

    if perfect == True or tail == "guard":
        tail_loop = InvalidCursor()
    else:
        tail_loop = outer_loop.next()

    outer_loop = proc.forward(outer_loop)
    inner_loop = proc.forward(inner_loop)
    if not isinstance(tail_loop, InvalidCursor):
        tail_loop = proc.forward(tail_loop)
    return proc, auto_divide_loop_cursors(outer_loop, inner_loop, tail_loop)


def scalar_to_simd(proc, loop_cursor, vec_width, memory_type, precision):
    """
    for i in seq(0, hi):
        lhs(i) = (e_0(i), e_1(i), ..., e_n(i));

    ----->

    for io in seq(0, hi / vec_width):
        reg0: precision[vec_width]
        for ii in seq(0, vec_width):
            reg0[ii] = e_0(ii)

        reg1: precision[vec_width]
        for ii in seq(0, vec_width):
            reg1[ii] = e_1(ii)

        ....

        regn: precision[vec_width]
        for ii in seq(0, vec_width):
            regn[ii] = e_n(ii)

        for ii in seq(0, vec_width):
            lhs(io * vec_width + ii) = e_n(ii);

    for i in seq(0, hi % vec_width):
        lhs(i + delta) = (e_0(i + delta), e_1(i + delta), ..., e_n(i + delta));
    """

    if not isinstance(loop_cursor, ForCursor):
        raise BLAS_SchedulingError("scalar_to_simd loop_cursor must be a ForCursor")

    loop_cursor = proc.forward(loop_cursor)

    if not (
        isinstance(loop_cursor.hi(), LiteralCursor)
        and loop_cursor.hi().value() == vec_width
    ):
        proc = divide_loop(
            proc,
            loop_cursor,
            vec_width,
            (loop_cursor.name() + "o", loop_cursor.name() + "i"),
            tail="cut",
        )

        outer_loop = proc.forward(loop_cursor)
        inner_loop = outer_loop.body()[0]
    else:
        inner_loop = loop_cursor

    inner_loop = proc.forward(inner_loop)

    stmts = []

    def fission_stmts(proc, body, depth=1):
        body_list = list(body)
        for stmt in body_list[:-1]:
            if isinstance(stmt, AllocCursor):
                proc = parallelize_and_lift_alloc(proc, stmt, n_lifts=depth)
                proc = set_memory(proc, stmt, memory_type)
                proc = set_precision(proc, stmt, precision)
            else:
                forwarded_stmt = proc.forward(stmt)
                stmts.append(stmt)
                proc = fission(proc, forwarded_stmt.after(), n_lifts=depth)
                forwarded_stmt = proc.forward(stmt)
                if isinstance(forwarded_stmt, IfCursor):
                    proc = fission_stmts(proc, forwarded_stmt.body(), depth + 1)
                elif isinstance(forwarded_stmt, ForCursor):
                    raise BLAS_SchedulingError("This is an inner loop vectorizer")
        forwarded_stmt = body_list[-1]
        stmts.append(forwarded_stmt)
        if isinstance(forwarded_stmt, IfCursor):
            proc = fission_stmts(proc, forwarded_stmt.body(), depth + 1)
        return proc

    proc = fission_stmts(proc, inner_loop.body())

    def detect_madd(expr):
        return (
            isinstance(expr, BinaryOpCursor)
            and expr.op() == "*"
            and isinstance(expr.parent(), ReduceCursor)
        )

    def get_expr_subtree_cursors(expr, stmt, alias):
        stmt_lhs_in_mem = False
        if isinstance(expr.parent(), StmtCursor):
            stmt_lhs_in_mem = (
                get_declaration(proc, stmt, stmt.name()).mem() == memory_type
            )

        if isinstance(expr, ReadCursor):
            expr_in_mem = get_declaration(proc, stmt, expr.name()).mem() == memory_type
            if alias or not (stmt_lhs_in_mem or expr_in_mem):
                return [expr]
            else:
                return []
        elif isinstance(expr, BinaryOpCursor):
            lhs = expr.lhs()
            rhs = expr.rhs()
            children_alias = (
                isinstance(lhs, ReadCursor)
                and isinstance(rhs, ReadCursor)
                and lhs.name() == rhs.name()
            )
            lhs_alias_with_stmt_lhs = False
            rhs_alias_with_stmt_lhs = False
            if isinstance(expr.parent(), StmtCursor):
                lhs_alias_with_stmt_lhs = (
                    isinstance(lhs, ReadCursor) and lhs.name() == expr.parent().name()
                )
                rhs_alias_with_stmt_lhs = (
                    isinstance(rhs, ReadCursor) and rhs.name() == expr.parent().name()
                )

            lhs_cursors = get_expr_subtree_cursors(
                lhs, stmt, lhs_alias_with_stmt_lhs or children_alias
            )
            rhs_cursors = get_expr_subtree_cursors(rhs, stmt, rhs_alias_with_stmt_lhs)

            if not detect_madd(expr) and not stmt_lhs_in_mem:
                return lhs_cursors + rhs_cursors + [expr]
            else:
                return lhs_cursors + rhs_cursors
        elif isinstance(expr, UnaryMinusCursor):
            return get_expr_subtree_cursors(expr.arg(), stmt, False) + [expr]
        elif isinstance(expr, BuiltInFunctionCursor):
            exprs = []
            for arg in expr.args():
                exprs = exprs + get_expr_subtree_cursors(arg, stmt, False)
            exprs = exprs + [expr]
            return exprs
        else:
            return [expr]

    is_assign_or_reduce = lambda stmt: isinstance(stmt, (AssignCursor, ReduceCursor))
    inner_loop = proc.forward(inner_loop)

    def vectorize_rhs(proc, stmt, depth=1):
        if not is_assign_or_reduce(stmt):
            return proc
        flat_rhs = get_expr_subtree_cursors(stmt.rhs(), stmt, False)

        name_gen = get_unique_names(proc)

        for expr in flat_rhs:
            proc = stage_expr(proc, expr, next(name_gen), precision, memory_type, depth)

        return proc

    def vectorize_stmt(proc, stmt, depth=1):
        proc = vectorize_rhs(proc, stmt, depth)
        if isinstance(stmt, ReduceCursor):
            alloc_stmt = get_declaration(proc, stmt, stmt.name())
            if alloc_stmt.mem() != memory_type:

                lhs_reg = next(get_unique_names(proc))
                proc = stage_mem(
                    proc, stmt, f"{stmt.name()}{expr_to_string(stmt.idx())}", lhs_reg
                )
                forwarded_stmt = proc.forward(stmt)

                alloc_cursor = forwarded_stmt.prev().prev()
                if depth > 1:
                    proc = lift_alloc(proc, alloc_cursor, n_lifts=depth - 1)
                proc = parallelize_and_lift_alloc(proc, alloc_cursor)

                forwarded_stmt = proc.forward(stmt)
                proc = fission(proc, forwarded_stmt.after(), n_lifts=depth)
                forwarded_stmt = proc.forward(forwarded_stmt)
                proc = fission(proc, forwarded_stmt.before(), n_lifts=depth)

                proc = set_memory(proc, alloc_cursor, memory_type)
                proc = set_precision(proc, alloc_cursor, precision)
        elif isinstance(stmt, IfCursor):
            assert len(stmt.body()) == 1
            if not isinstance(stmt.orelse(), InvalidCursor):
                raise BLAS_SchedulingError("Not implemented yet")
            proc = vectorize_stmt(proc, stmt.body()[0], depth + 1)
        elif isinstance(stmt, AssignCursor):
            # This will be a store
            pass
        else:
            raise BLAS_SchedulingError("Not implemented yet")
        return proc

    for stmt in stmts:
        forwarded_stmt = proc.forward(stmt)
        if isinstance(forwarded_stmt, IfCursor):
            continue
        inner_loop = get_enclosing_loop(proc, forwarded_stmt)
        if isinstance(inner_loop, ForCursor):
            assert len(inner_loop.body()) == 1
            proc = vectorize_stmt(proc, inner_loop.body()[0])

    return proc


def interleave_loop(proc, loop, factor=None, par_reduce=False, memory=DRAM, tail="cut"):
    """
    for i in seq(0, c):
        S1
        S2
        S3

    ----->
    s1 x c
    s2 x c
    s3 x c
    """

    loop = proc.forward(loop)

    def rewrite(proc, loop, factor=None, par_reduce=False, memory=DRAM, tail="cut"):
        loop = proc.forward(loop)
        if factor is not None:
            proc, (outer, loop, _) = auto_divide_loop(proc, loop, factor, tail=tail)
            if par_reduce:
                proc = parallelize_all_reductions(
                    proc, outer, memory=memory, unroll=True
                )
                loop = proc.forward(outer).body()[0]
        else:
            if par_reduce:
                proc = parallelize_all_reductions(
                    proc, loop, memory=memory, unroll=True
                )

        allocs = filter(lambda s: isinstance(s, AllocCursor), loop.body())
        proc = apply(parallelize_and_lift_alloc)(proc, allocs)

        stmts = list(proc.forward(loop).body())
        proc = apply(fission)(proc, [s.after() for s in stmts[:-1]])
        proc = apply(unroll_loop)(proc, [proc.forward(s).parent() for s in stmts])
        return proc

    if tail in {"cut", "cut_and_guard"}:
        proc = rewrite(proc, loop, factor, par_reduce, memory, tail)
    elif tail == "recursive":
        if factor is None:
            raise BLAS_SchedulingError(
                "Cannot specify recursive tail strategy and factor=None"
            )
        proc, (_, inners, _) = divide_loop_recursive(
            proc, loop, factor, tail="cut", rc=True
        )
        proc = apply(rewrite)(proc, inners, par_reduce=par_reduce, memory=memory)
    elif tail == "specialize":
        if factor is None:
            raise BLAS_SchedulingError(
                "Cannot specify recursive tail strategy and factor=None"
            )
        proc = rewrite(proc, loop, factor, par_reduce, memory, tail="cut")
        tail_loop = proc.forward(loop).next()
        proc, (stmts,) = binary_specialize(
            proc, tail_loop, tail_loop.hi(), [i for i in range(factor)], rc=True
        )
        proc = apply(rewrite)(proc, stmts, par_reduce=par_reduce, memory=memory)
    else:
        raise BLAS_SchedulingError(f"Unknown tail strategy: {tail}")
    return proc


@dataclass
class hoist_stmt_cursors:
    allocs: list[AllocCursor]
    stmt: StmtCursor
    loop: ForCursor

    def __iter__(self):
        yield self.allocs
        yield self.stmt
        yield self.loop


def hoist_stmt(proc, stmt, rc=False):
    """
    for i in seq(0, hi):
        B1;
        s;
        B2;

    --->

    s;
    for i in seq(0, hi):
        B1;
        B2;
    """
    stmt = proc.forward(stmt)

    # Type Checking
    if not isinstance(stmt, StmtCursor):
        raise BLAS_SchedulingError("Cannot hoist cursor that are not statements")

    loop = stmt.parent()

    # Pre-condition 1: a scope exists
    if not isinstance(loop, ForCursor):
        raise BLAS_SchedulingError("Statement is not within a loop")

    # Pre-condition 2: fail-fast, no dependency on a loop
    deps = list(get_symbols(proc, stmt))
    if isinstance(loop, ForCursor) and loop.name() in deps:
        raise BLAS_SchedulingError(
            "Cannot hoist cursor to a statement that depends on enclosing loop"
        )

    # Alloc is a special case
    if isinstance(stmt, AllocCursor):
        proc = lift_alloc(proc, stmt)
        if not rc:
            return proc
        loop = proc.forward(loop)
        stmt = proc.forward(stmt)
        return proc, hoist_stmt_cursors([stmt], stmt, loop)

    allocs = []

    # Reorder the statement to the top of the loop
    while not isinstance(stmt.prev(), InvalidCursor):
        prev_stmt = stmt.prev()
        if isinstance(prev_stmt, AllocCursor) and prev_stmt.name() in deps:
            proc = lift_alloc(proc, prev_stmt)
            allocs.append(prev_stmt)
        else:
            proc = reorder_stmts(proc, stmt.expand(1, 0))
        stmt = proc.forward(stmt)

    # Pull the statement on its own outside the loop
    if len(loop.body()) > 1:
        proc, (_, loop) = fission_(proc, stmt.after(), rc=True)
        stmt = proc.forward(stmt)
    proc = remove_loop(proc, stmt.parent())

    if not rc:
        return proc

    allocs = [proc.forward(a) for a in allocs]
    stmt = proc.forward(stmt)
    loop = proc.forward(loop)
    return proc, hoist_stmt_cursors(allocs, stmt, loop)


@dataclass
class hoist_from_loop_cursors:
    hoisted: list
    loop: ForCursor

    def __iter__(self):
        yield self.hoisted
        yield self.loop


def hoist_from_loop(proc, loop, rc=False):
    loop = proc.forward(loop)

    if not is_loop(proc, loop):
        raise BLAS_SchedulingError(f"loop must of type {ForCursor} not {type(loop)}")

    rcs = []

    def hoist_non_alloc(proc, stmt):
        stmt = proc.forward(stmt)
        if isinstance(stmt, AllocCursor):
            return proc
        proc, cursors = hoist_stmt(proc, stmt, rc=True)
        rcs.append(cursors)
        return proc

    proc = apply(attempt(hoist_non_alloc))(proc, loop.body())

    if not rc:
        return proc

    if not rcs:
        return proc, hoist_from_loop_cursors([], loop)

    loop = rcs[-1].loop
    stmt_allocs = []
    for cursors in rcs:
        stmt = proc.forward(cursors.stmt)
        allocs = tuple(proc.forward(alloc) for alloc in cursors.allocs)
        stmt_allocs.append((stmt, allocs))
    return proc, hoist_from_loop_cursors(stmt_allocs, loop)


@dataclass
class jam_stmt_cursors:
    loop: ForCursor
    stmt: StmtCursor

    def __iter__(self):
        yield self.loop
        yield self.stmt


def jam_stmt(proc, stmt, unsafe_disable_check=False, rc=False):
    stmt = proc.forward(stmt)
    loop = stmt.next()
    if not is_loop(proc, loop):
        raise BLAS_SchedulingError("Next statement must be a loop.")

    proc = add_loop(
        proc, stmt, loop.name(), FormattedExprStr("_ - _", loop.hi(), loop.lo())
    )
    stmt = proc.forward(stmt)
    stmt_loop = proc.forward(stmt).parent()
    proc = shift_loop(proc, stmt_loop, FormattedExprStr("_", loop.lo()))
    proc = simplify(proc)
    proc = fuse(proc, stmt_loop, loop, unsafe_disable_check=unsafe_disable_check)
    proc = repeate(reorder_stmt_forward)(proc, stmt)

    if not rc:
        return proc
    stmt = proc.forward(stmt)
    stmt_loop = proc.forward(stmt_loop)
    return proc, jam_stmt_cursors(stmt_loop, stmt)


def parallelize_reduction(
    proc, reduce_stmt, factor=None, memory=DRAM, nth_loop=1, unroll=False
):
    # Auto-coersion
    if isinstance(unroll, bool):
        unroll = (unroll, unroll)

    reduce_stmt = proc.forward(reduce_stmt)

    if not is_reduce(proc, reduce_stmt):
        raise TypeError(f"reduce_stmt must of type {ReduceCursor}")

    reduce_loop = get_enclosing_loop(proc, reduce_stmt, nth_loop)

    if factor is not None:
        proc, (reduce_loop, _, _) = auto_divide_loop(
            proc, reduce_loop, factor, tail="guard"
        )
        proc = simplify(proc)
        nth_loop += 1

    # Stage reduction around the loop we are reducing over
    proc = reorder_loops(proc, reduce_loop)
    proc = auto_stage_mem(
        proc,
        reduce_stmt,
        next(get_unique_names(proc)),
        n_lifts=nth_loop - 1,
        accum=True,
    )
    proc = simplify(proc)

    # Set the memory of the newly created buffer
    reduce_loop = proc.forward(reduce_loop)
    alloc = reduce_loop.prev().prev()
    proc = set_memory(proc, alloc, memory)

    # Parallelize the reduction
    reduce_loop = proc.forward(reduce_loop)
    proc = parallelize_and_lift_alloc(proc, reduce_loop.prev().prev())

    # Fission the zero and store-back stages
    proc = fission(proc, reduce_loop.before())
    proc = fission(proc, reduce_loop.after())

    # Reorder the loop nest back
    reduce_loop = proc.forward(reduce_loop)
    proc = reorder_loops(proc, reduce_loop.parent())

    # Unroll any loops
    reduce_loop = proc.forward(reduce_loop)
    if unroll[0]:
        proc = unroll_loop(proc, reduce_loop.prev())
    if unroll[1]:
        proc = unroll_loop(proc, reduce_loop.next())

    if factor is not None:
        proc = undo_divide_and_guard_loop(proc, reduce_loop)
    return proc


def parallelize_all_reductions(proc, loop, factor=None, memory=DRAM, unroll=False):
    loop = proc.forward(loop)

    def rewrite(proc, s):
        s = proc.forward(s)
        nth_loop = 0
        for parent in get_parents(proc, s):
            if is_loop(proc, parent):
                nth_loop += 1
            if parent == loop:
                break
        return parallelize_reduction(proc, s, factor, memory, nth_loop, unroll)

    return make_pass(attempt(rewrite))(proc, loop)


def unroll_and_jam(proc, loop, factor, unroll=(True, True, True)):
    inner_loops = [i for i in loop.body() if isinstance(i, ForCursor)]
    if len(inner_loops) > 1:
        raise BLAS_SchedulingError("Multiple loops found, decision is ambigious")
    if len(inner_loops) == 0:
        raise BLAS_SchedulingError("No loops found")

    return interleave_outer_loop_with_inner_loop(
        proc, loop, inner_loops[0], factor, unroll=unroll
    )


def unroll_and_jam_parent(proc, loop, factor, unroll=(True, True, True)):
    outer_loop = loop.parent()
    if not isinstance(outer_loop, ForCursor):
        raise BLAS_SchedulingError("parent is not a loop")
    return interleave_outer_loop_with_inner_loop(
        proc, outer_loop, loop, factor, unroll=unroll
    )


def interleave_outer_loop_with_inner_loop(
    proc,
    outer_loop_cursor,
    inner_loop_cursor,
    interleave_factor,
    unroll=(True, True, True),
):
    # TODO: check if inner_loop is directly in the body of outer_loop
    outer_loop_cursor = proc.forward(outer_loop_cursor)
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if (
        isinstance(outer_loop_cursor.hi(), LiteralCursor)
        and outer_loop_cursor.hi().value() == interleave_factor
    ):
        middle_loop_cursor = outer_loop_cursor
    else:
        proc = divide_loop(
            proc,
            outer_loop_cursor,
            interleave_factor,
            (outer_loop_cursor.name() + "o", outer_loop_cursor.name() + "i"),
            tail="cut",
        )

        outer_loop_cursor = proc.forward(outer_loop_cursor)
        middle_loop_cursor = outer_loop_cursor.body()[0]
    middle_loop_stmts = list(middle_loop_cursor.body())

    proc = simplify(proc)
    for stmt in middle_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if not isinstance(inner_loop_cursor.prev(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.before())
        inner_loop_cursor = proc.forward(inner_loop_cursor)
        if unroll[0]:
            proc = unroll_loop(proc, inner_loop_cursor.parent().prev())

    if not isinstance(inner_loop_cursor.next(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.after())
        inner_loop_cursor = proc.forward(inner_loop_cursor)
        if unroll[2]:
            proc = unroll_loop(proc, inner_loop_cursor.parent().next())

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    proc = reorder_loops(proc, inner_loop_cursor.parent())
    if unroll[1]:
        proc = unroll_loop(proc, inner_loop_cursor.parent())

    return proc


def cut_tail_and_unguard(proc, loop):
    loop = proc.forward(loop)

    # Cut the tail iterations
    last_outer_iteration = FormattedExprStr("_ - 1", loop.hi())
    proc = cut_loop(proc, loop, last_outer_iteration)

    # Unguard
    proc = dce(proc, loop)

    return proc


def parallelize_allocs(proc, cursor):
    if not isinstance(cursor, (ForCursor, IfCursor)):
        raise BLAS_SchedulingError(
            f"Got type {type(cursor)}, expected {ForCursor} or {IfCursor}"
        )

    allocs = filter(lambda s: isinstance(s, AllocCursor), nlr_stmts(proc, cursor))
    func = lambda proc, alloc: parallelize_and_lift_alloc(
        proc, alloc, get_distance(proc, alloc, cursor)
    )
    return apply(func)(proc, allocs)


def fission_into_singles(proc, cursor):
    if not isinstance(cursor, (ForCursor, IfCursor)):
        raise BLAS_SchedulingError(
            f"Got type {type(cursor)}, expected {ForCursor} or {IfCursor}"
        )

    cursor = proc.forward(cursor)

    def dfs(proc, cursor, n_lifts=0):
        if n_lifts and not is_end_of_body(proc, cursor):
            proc = fission(proc, cursor.after(), n_lifts)
        children = get_children(proc, cursor)
        children = filter(lambda s: isinstance(s, StmtCursor), children)
        return apply(dfs)(proc, children, n_lifts + 1)

    proc = parallelize_allocs(proc, cursor)
    return dfs(proc, cursor)


@dataclass
class divide_and_predicate_stmts_cursors:
    outer_loop: ForCursor
    inner_loop: ForCursor

    def __iter__(self):
        yield self.outer_loop
        yield self.inner_loop


def divide_and_predicate_stmts(proc, loop, factor, rc=True):
    proc, (hoisted, loop) = hoist_from_loop(proc, loop, rc=True)
    proc, (outer, inner, _) = auto_divide_loop(proc, loop, factor, tail="guard")
    proc = simplify(proc)
    proc = fission_into_singles(proc, inner.body()[0])
    for (stmt, allocs) in hoisted[::-1]:
        proc, (outer, _) = jam_stmt(proc, stmt, unsafe_disable_check=True, rc=True)
        proc, (inner, _) = jam_stmt(proc, stmt, unsafe_disable_check=True, rc=True)
        proc = apply(sink_alloc)(proc, allocs)
        proc = apply(sink_alloc)(proc, allocs)
    if not rc:
        return proc
    outer = proc.forward(outer)
    inner = proc.forward(inner)
    return proc, divide_and_predicate_stmts_cursors(outer, inner)


@dataclass
class vectorize_cursors:
    loop: ForCursor

    def __iter__(self):
        yield self.loop


def vectorize_predicate_tail(
    proc,
    loop,
    vec_width,
    precision,
    mem_type,
    instructions=[],
    tail="cut_and_predicate",
    rc=False,
):

    proc = parallelize_all_reductions(proc, loop, factor=vec_width, memory=mem_type)

    allocs = filter(lambda s: isinstance(s, AllocCursor), nlr_stmts(proc, loop))
    proc = apply(set_memory)(proc, allocs, mem_type)

    children_ops = [fma_rule, abs_rule]
    proc = stage_compute(proc, loop, precision, mem_type, children_ops)

    proc, (outer, inner) = divide_and_predicate_stmts(proc, loop, vec_width, rc=True)
    proc = simplify(proc)
    proc = fission_into_singles(proc, inner)

    if tail == "cut_and_predicate":
        proc = cut_tail_and_unguard(proc, outer)

    proc = replace_all_stmts(proc, instructions)

    if not rc:
        return proc
    outer = proc.forward(outer)
    return proc, vectorize_cursors(outer)


def vectorize(
    proc,
    loop,
    vec_width,
    precision,
    mem_type,
    instructions=[],
    tail="cut_and_predicate",
    rc=True,
):
    if tail in {"predicate", "cut_and_predicate"}:
        return vectorize_predicate_tail(
            proc, loop, vec_width, precision, mem_type, instructions, tail, rc
        )

    # Tile to exploit vectorization
    proc, (outer, inner, _) = auto_divide_loop(proc, loop, vec_width, tail=tail)

    proc = parallelize_all_reductions(proc, outer, memory=mem_type)

    # Previous step calls fission which would change what
    # inner loop we are pointing at
    outer = proc.forward(outer)
    inner = outer.body()[0]

    allocs = filter(lambda s: isinstance(s, AllocCursor), nlr_stmts(proc, inner))
    proc = apply(set_memory)(proc, allocs, mem_type)

    children_ops = [fma_rule, abs_rule]
    proc = stage_compute(proc, inner, precision, mem_type, children_ops)
    proc = fission_into_singles(proc, inner)

    proc = replace_all_stmts(proc, instructions)
    if not rc:
        return proc
    outer = proc.forward(outer)
    return proc, vectorize_cursors(outer)


def tile_loops_top_down(proc, loop_tile_pairs):

    loop_tile_pairs = [(proc.forward(i[0]), i[1]) for i in loop_tile_pairs]

    inner_loops = []
    for i in range(len(loop_tile_pairs)):
        outer_loop = loop_tile_pairs[i][0]
        tile_size = loop_tile_pairs[i][1]
        new_names = (outer_loop.name() + "o", outer_loop.name() + "i")
        if isinstance(outer_loop.hi(), LiteralCursor) and (
            outer_loop.hi().value() % tile_size == 0
        ):
            proc = divide_loop(proc, outer_loop, tile_size, new_names, perfect=True)
        else:
            proc = divide_loop(proc, outer_loop, tile_size, new_names, tail="cut")
        inner_loop = proc.forward(outer_loop).body()[0]
        inner_loops.append(inner_loop)

    for i in range(len(loop_tile_pairs) - 2, -1, -1):
        inner_loop = inner_loops[i]
        tile_size = loop_tile_pairs[i][1]
        for j in range(i + 1, len(loop_tile_pairs)):
            loop = loop_tile_pairs[j][0]
            proc = interleave_outer_loop_with_inner_loop(
                proc, inner_loop, loop, tile_size, (False, False, False)
            )
    return proc, [proc.forward(l) for l in inner_loops]


def tile_loops_bottom_up(proc, outer_most_loop, tiles):
    loop = outer_most_loop
    for i in tiles[:-1]:
        if not len(loop.body()) == 1:
            raise BLAS_SchedulingError("All loop must have a body length of 1")
        if not isinstance(loop.body()[0], ForCursor):
            raise BLAS_SchedulingError("Did not find a nested loop")

    loops = []
    loop = outer_most_loop
    for i in tiles:
        loops.append((loop, i))
        loop = loop.body()[0]

    def get_depth(loop):
        if not isinstance(loop, ForCursor):
            return 0
        return max([get_depth(i) for i in loop.body()]) + 1

    def push_loop_in(proc, loop, depth):
        if get_depth(loop) == depth:
            return proc
        count = len(loop.body())
        for stmt in list(loop.body())[:-1]:
            proc = fission(proc, stmt.after())
        loop = proc.forward(loop)
        loops = []
        for i in range(count):
            loops.append(loop)
            loop = loop.next()
        for loop in loops:
            if get_depth(loop) == depth:
                continue
            proc = reorder_loops(proc, loop)
            proc = push_loop_in(proc, proc.forward(loop), depth)
        return proc

    for depth, (loop, tile) in enumerate(loops[::-1]):
        proc, cursors = auto_divide_loop(proc, loop, tile, tail="cut")
        proc = push_loop_in(proc, cursors.inner_loop, depth + 1)
        proc = push_loop_in(proc, cursors.tail_loop, depth + 1)

    return proc


def auto_stage_mem(proc, cursor, new_buff_name=None, n_lifts=1, accum=False):
    if not isinstance(cursor, (ReadCursor, ReduceCursor, AssignCursor)):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    if new_buff_name is None:
        new_buff_name = next(get_unique_names(proc))

    cursor = proc.forward(cursor)

    lo = []
    hi = []
    loops = []
    for n in range(1, n_lifts + 1):
        loop = get_enclosing_loop(proc, cursor, n)
        loops.append(loop)

    subst = {}
    for i in range(len(loops) - 1, -1, -1):
        loop = loops[i]
        subst[loop.name()] = f"(({expr_to_string(loop.hi(), subst)})-1)"

    for idx in cursor.idx():
        hi.append(expr_to_string(idx, subst))

    for key in subst:
        subst[key] = 0

    for idx in cursor.idx():
        lo.append(expr_to_string(idx, subst))

    def ith_idx(i):
        if lo[i] == hi[i]:
            return lo[i]
        else:
            return f"{lo[i]}:(({hi[i]})+1)"

    window = ",".join([ith_idx(i) for i in range(len(cursor.idx()))])
    window = f"{cursor.name()}[{window}]" if window else cursor.name()
    block = cursor if n_lifts == 0 else loops[-1]
    block = block if isinstance(block, StmtCursor) else get_enclosing_stmt(proc, block)
    return stage_mem(proc, block, window, new_buff_name, accum=accum)


def ordered_stage_expr(proc, expr_cursors, new_buff_name, precision, n_lifts=1):
    if not isinstance(expr_cursors, list):
        expr_cursors = [expr_cursors]

    if not all([isinstance(cursor, ExprCursor) for cursor in expr_cursors]):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    expr_cursors = [proc.forward(c) for c in expr_cursors]
    original_stmt = get_statement(expr_cursors[0])

    proc = bind_expr(proc, expr_cursors, new_buff_name)
    original_stmt = proc.forward(original_stmt)
    assign_cursor = original_stmt.prev()
    alloc_cursor = assign_cursor.prev()
    expr_cursor = assign_cursor.rhs()
    deps = list(get_symbols(proc, expr_cursor))

    assert isinstance(assign_cursor, AssignCursor)
    assert isinstance(alloc_cursor, AllocCursor)

    anchor_stmt = assign_cursor

    def hoist_as_loop(proc, stmt_cursor):
        stmt_cursor = proc.forward(stmt_cursor)
        while not isinstance(stmt_cursor.prev(), InvalidCursor):
            proc = reorder_stmts(proc, stmt_cursor.expand(1, 0))
            stmt_cursor = proc.forward(stmt_cursor)

        proc = fission(proc, stmt_cursor.after())

        return proc

    for i in range(n_lifts):
        parent = anchor_stmt.parent()

        if not isinstance(parent, ForCursor):
            raise BLAS_SchedulingError("Not implemented yet")
        if parent.name() in deps:
            proc = parallelize_and_lift_alloc(proc, alloc_cursor)
        else:
            proc = lift_alloc(proc, alloc_cursor)

        proc = hoist_as_loop(proc, anchor_stmt)
        anchor_stmt = proc.forward(anchor_stmt)
        anchor_stmt = anchor_stmt.parent()

    alloc_cursor = proc.forward(alloc_cursor)
    loop_nest = alloc_cursor.next()

    def try_removing_loops(proc, loop):
        child_stmt = loop.body()[0]
        if isinstance(child_stmt, ForCursor):
            proc = try_removing_loops(proc, child_stmt)
        try:
            proc = remove_loop(proc, loop)
        except:
            pass
        return proc

    proc = try_removing_loops(proc, loop_nest)
    alloc_cursor = proc.forward(alloc_cursor)
    proc = set_precision(proc, alloc_cursor, precision)
    scopes_nest = alloc_cursor.next()

    def lift_all_ifs(proc, scope, depth=0):
        if isinstance(scope, IfCursor):
            for i in range(depth):
                proc = lift_scope(proc, scope)
        child_stmt = scope.body()[0]
        if isinstance(child_stmt, (ForCursor, IfCursor)):
            proc = lift_all_ifs(proc, child_stmt, depth + 1)
        return proc

    proc = lift_all_ifs(proc, scopes_nest)

    return proc


def _eliminate_dead_code_pruned(proc, s):
    s = proc.forward(s)
    if isinstance(s, ForCursor) and is_loop_bounds_const(s):
        return proc
    else:
        return eliminate_dead_code(proc, s)


dce = make_pass(attempt(_eliminate_dead_code_pruned))


def unroll_buffers(proc, block=InvalidCursor(), mem=None):
    def rewrite(proc, alloc):
        alloc = proc.forward(alloc)
        if not isinstance(alloc, AllocCursor):
            return proc
        if not alloc.is_tensor():
            return proc
        diff = int(alloc.mem() is mem)
        for i in range(0, len(alloc.shape()) - diff):
            if isinstance(alloc.shape()[i], LiteralCursor):
                return unroll_buffer(proc, alloc, i)
        return proc

    return make_pass(rewrite)(proc, block)


def unfold_reduce(proc, reduce):
    if not isinstance(reduce, ReduceCursor):
        raise BLAS_SchedulingError("Expected a reduce cursor")

    proc = auto_stage_mem(proc, reduce, n_lifts=0)
    reduce = proc.forward(reduce)
    alloc = reduce.prev().prev()
    proc = merge_writes(proc, reduce.as_block().expand(delta_lo=1, delta_hi=0))
    assign = proc.forward(alloc).next()
    proc = inline_assign(proc, assign)
    proc = delete_buffer(proc, alloc)

    return proc


def fma_rule(proc, expr):
    expr = proc.forward(expr)

    if is_add(proc, expr):
        if is_mul(proc, expr.lhs()):
            # (a * b) + c
            return [expr.lhs().lhs(), expr.lhs().rhs(), expr.rhs()]
        elif is_mul(proc, expr.rhs()):
            # a + (b * c)
            return [expr.lhs(), expr.rhs().lhs(), expr.rhs().rhs()]

    return None


def abs_rule(proc, expr):
    expr = proc.forward(expr)
    if is_select(proc, expr):
        args = expr.args()
        if (
            is_literal(proc, args[0], 0.0)
            and is_unary_minus(proc, args[3])
            and are_exprs_equal(proc, args[1], args[2])
            and are_exprs_equal(proc, args[1], args[3].arg())
        ):
            return [[args[1], args[2], args[3].arg()]]
    return None


def stage_expr_into_memory(proc, exprs, precision, memory):
    if not isinstance(exprs, list):
        exprs = [exprs]

    expr = proc.forward(exprs[0])

    # No need to stage if expr is already assigned
    # to the target memory
    parent = expr.parent()
    if (
        isinstance(parent, AssignCursor)
        and get_declaration(proc, expr, parent.name()).mem() is memory
    ):
        return proc, expr

    # No need to stage if expr is already read
    # from the target memory
    if (
        isinstance(expr, ReadCursor)
        and get_declaration(proc, expr, expr.name()).mem() is memory
    ):
        return proc, expr

    return lift_rc(bind_and_set_expr, "bound_expr")(proc, exprs, precision, memory)


def stage_compute(
    proc,
    block=InvalidCursor(),
    precision="R",
    memory=DRAM,
    children_ops=[],
):

    if not isinstance(children_ops, list):
        raise BLAS_SchedulingError("Expected children_ops to be a list")

    def get_numeric_children(proc, cursor=InvalidCursor()):
        check = lambda c: hasattr(c, "type") and c.type().is_numeric()
        yield from filter(check, get_children(proc, cursor))

    children_ops.append(get_numeric_children)

    def stage(proc, exprs):
        proc, expr = stage_expr_into_memory(proc, exprs, precision, memory)
        for children_op in children_ops:
            if children := children_op(proc, expr):
                break
        return apply(stage)(proc, children)

    proc = make_pass(attempt(unfold_reduce))(proc, block)
    assigns = filter(lambda s: isinstance(s, AssignCursor), lrn_stmts(proc, block))
    exprs = [assign.rhs() for assign in assigns]
    proc = apply(stage)(proc, exprs)
    proc = make_pass(attempt(fold_into_reduce))(proc, block)
    return proc


def replace_all_stmts(proc, instructions):
    if not isinstance(instructions, list):
        instructions = [instructions]

    for stmt in nlr_stmts(proc):
        try:
            stmt = proc.forward(stmt)
        except InvalidCursorError:
            continue

        for instr in instructions:
            try:
                proc = call_site_mem_aware_replace(proc, stmt, instr, quiet=True)
                break
            except:
                pass
    return proc


def bound_loop_by_if(proc, loop):
    loop = proc.forward(loop)
    err = "Expected loop to be of the following structure:\nfor iter in seq(lo, hi):\n\t if iter < e:"
    if len(loop.body()) != 1 or not isinstance(loop.body()[0], IfCursor):
        raise BLAS_SchedulingError(err)

    if_c = loop.body()[0]
    if not isinstance(if_c.orelse(), InvalidCursor):
        raise BLAS_SchedulingError(err)

    if (
        not isinstance(if_c.cond().lhs(), ReadCursor)
        or if_c.cond().lhs().name() != loop.name()
        or if_c.cond().op() != "<"
    ):
        raise BLAS_SchedulingError(err)

    if_c = loop.body()[0]
    proc = cut_loop(proc, loop, FormattedExprStr("_ + _", loop.lo(), if_c.cond().rhs()))
    loop1 = proc.forward(loop)
    loop2 = loop1.next()
    proc = eliminate_dead_code(proc, loop1.body()[0])
    proc = eliminate_dead_code(proc, loop2.body()[0])
    # proc = delete_pass(proc, loop2), but it doesn't forward it
    return proc


def undo_divide_and_guard_loop(proc, loop):
    loop = proc.forward(loop)
    proc = mult_loops(proc, loop, loop.name()[:-1])
    proc = simplify(proc)
    proc = bound_loop_by_if(proc, loop)
    return proc


def cleanup(proc):
    proc = simplify(proc)
    proc = dce(proc)
    try:
        proc.find("pass")
        proc = delete_pass(proc)
    except SchedulingError:
        pass
    return proc


def reorder_stmt_forward(proc, stmt):
    stmt = proc.forward(stmt)
    block = stmt.as_block().expand(0, 1)
    return reorder_stmts(proc, block)


def reorder_stmt_backwards(proc, stmt):
    stmt = proc.forward(stmt)
    block = stmt.as_block().expand(-1, 0)
    return reorder_stmts(proc, block)


@dataclass
class divide_loop_recursive_cursors:
    outer_loops: list
    inner_loops: list
    tail_loop: ForCursor

    def __iter__(self):
        yield self.outer_loops
        yield self.inner_loops
        yield self.tail_loop


def divide_loop_recursive(proc, loop, factor, tail="cut", rc=False):
    if tail not in {"cut", "cut_and_guard"}:
        raise BLAS_SchedulingError("tail strategy must be cut or cut_and_guard")
    outer_loops = []
    inner_loops = []
    tail_loop = loop
    while factor > 1:
        proc, (outer, inner, tail_loop) = auto_divide_loop(
            proc, tail_loop, factor, tail=tail
        )
        outer_loops.append(outer)
        inner_loops.append(inner)
        factor = factor // 2
    if not rc:
        return proc
    outer_loops = [proc.forward(c) for c in outer_loops]
    inner_loops = [proc.forward(c) for c in inner_loops]
    return proc, divide_loop_recursive_cursors(outer_loops, inner_loops, tail_loop)


@dataclass
class specialize_cursors:
    if_stmt: Cursor

    def __iter__(self):
        yield self.if_stmt


def specialize_(proc, stmt, cond, rc=False):
    stmt = proc.forward(stmt)
    parent = stmt.parent()
    index = get_index_in_body(proc, stmt)
    proc = specialize(proc, stmt, cond)
    if not rc:
        return proc
    is_else = False
    if (
        isinstance(parent, IfCursor)
        and index < len(parent.orelse())
        and parent.orelse()[index] == stmt
    ):
        is_else = True
    if not isinstance(parent, InvalidCursor):
        parent = proc.forward(parent)
    else:
        parent = proc

    if_stmt = parent.body()[index] if not is_else else parent.orelse()[index]
    return proc, specialize_cursors(if_stmt)


@dataclass
class binary_specialize_cursors:
    stmts: Cursor

    def __iter__(self):
        yield self.stmts


def binary_specialize(proc, stmt, expr, values, rc=False):
    stmt = proc.forward(stmt)
    if isinstance(expr, ExprCursor):
        expr = proc.forward(expr)
        expr = expr_to_string(expr)
    get_cond = lambda op, v: f"{expr} {op} {v}"

    if len(values) == 1:
        raise BLAS_SchedulingError("Cannot specialize given one value!")
    values = sorted(values)
    stmt = proc.forward(stmt)

    stmts = []

    def rewrite(proc, stmt, values):
        if len(values) == 1:
            # This should be redundant if the user provided correct inputs!
            # So, it is really a check that the inputs the user provided cover the full range.
            proc, (if_stmt,) = specialize_(
                proc, stmt, get_cond("==", values[0]), rc=True
            )
            proc = simplify(proc)
            proc = eliminate_dead_code(proc, if_stmt)
            stmts.append(if_stmt.body()[0])
            stmts.append(if_stmt.orelse()[0])
            return proc
        md = len(values) // 2
        proc, (if_stmt,) = specialize_(proc, stmt, get_cond("<", values[md]), rc=True)
        proc = rewrite(proc, if_stmt.body()[0], values[:md])
        proc = rewrite(proc, if_stmt.orelse()[0], values[md:])
        return proc

    proc = rewrite(proc, stmt, values)
    if not rc:
        return proc

    filtered_stmts = []
    for s in stmts:
        try:
            stmt = proc.forward(s)
            if not isinstance(stmt, PassCursor):
                filtered_stmts.append(stmt)
        except InvalidCursorError:
            pass
    return proc, binary_specialize_cursors(filtered_stmts)
