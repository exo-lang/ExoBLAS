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
    elif isinstance(expr_cursor, BuiltInCursor):
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
    outer_loop_cursor: ForCursor
    inner_loop_cursor: ForCursor
    tail_loop_cursor: ForCursor


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
    outer_loop_cursor = proc.forward(loop_cursor)
    inner_loop_cursor = outer_loop_cursor.body()[0]

    if perfect == True or tail == "guard":
        tail_loop_cursor = InvalidCursor()
    else:
        tail_loop_cursor = outer_loop_cursor.next()

    return proc, auto_divide_loop_cursors(
        outer_loop_cursor, inner_loop_cursor, tail_loop_cursor
    )


def scalar_loop_to_simd_loops(proc, loop_cursor, vec_width, memory_type, precision):
    return vectorize_to_loops(proc, loop_cursor, vec_width, memory_type, precision)


def vectorize_to_loops(proc, loop_cursor, vec_width, memory_type, precision):
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
        raise BLAS_SchedulingError("vectorize_to_loops loop_cursor must be a ForCursor")

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

        outer_loop_cursor = proc.forward(loop_cursor)
        inner_loop_cursor = outer_loop_cursor.body()[0]
    else:
        inner_loop_cursor = loop_cursor

    inner_loop_cursor = proc.forward(inner_loop_cursor)

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

    proc = fission_stmts(proc, inner_loop_cursor.body())

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
    inner_loop_cursor = proc.forward(inner_loop_cursor)

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


def interleave_execution(proc, loop_cursor, interleave_factor):
    """
    for i in seq(0, n):
        S1
        S2
        S3

    ----->

    for io in seq(0, n / interleave_factor):
        S1
        ... x interleave_factor
        S2
        ... x interleave_factor
        S3
        ... x interleave_factor
    """
    if not isinstance(loop_cursor, ForCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForCursor")

    if interleave_factor == 1:
        return proc

    loop_cursor = proc.forward(loop_cursor)

    if not (
        isinstance(loop_cursor.hi(), LiteralCursor)
        and loop_cursor.hi().value() == interleave_factor
    ):
        proc = divide_loop(
            proc,
            loop_cursor,
            interleave_factor,
            (loop_cursor.name() + "o", loop_cursor.name() + "i"),
            tail="cut",
        )

        outer_loop_cursor = proc.forward(loop_cursor)
        inner_loop_cursor = outer_loop_cursor.body()[0]
    else:
        inner_loop_cursor = loop_cursor

    inner_loop_stmts = list(inner_loop_cursor.body())

    for stmt in inner_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    inner_loop_stmts = list(inner_loop_cursor.body())

    for stmt in inner_loop_stmts[:-1]:
        forwarded_stmt = proc.forward(stmt)
        proc = fission(proc, forwarded_stmt.after())
        forwarded_stmt = proc.forward(stmt).parent()
        proc = unroll_loop(proc, forwarded_stmt)

    last_loop = proc.forward(inner_loop_stmts[-1]).parent()
    proc = unroll_loop(proc, last_loop)

    return proc


def hoist_stmt(proc, stmt):
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
        return lift_alloc(proc, stmt)

    # Reorder the statement to the top of the loop
    while not isinstance(stmt.prev(), InvalidCursor):
        prev_stmt = stmt.prev()
        if isinstance(prev_stmt, AllocCursor) and prev_stmt.name() in deps:
            proc = lift_alloc(proc, prev_stmt)
        else:
            proc = reorder_stmts(proc, stmt.expand(1, 0))
        stmt = proc.forward(stmt)

    # Pull the statement on its own outside the loop
    if len(loop.body()) > 1:
        proc = fission(proc, stmt.after())
        stmt = proc.forward(stmt)
    proc = remove_loop(proc, stmt.parent())
    return proc


def apply_to_block(proc, block_cursor, stmt_scheduling_op):
    if not isinstance(block_cursor, BlockCursor):
        raise BLAS_SchedulingError("cannot apply to a non-block cursor")

    for stmt in block_cursor:
        try:
            proc = stmt_scheduling_op(proc, stmt)
        except (SchedulingError, BLAS_SchedulingError):
            pass

    return proc


def parallelize_reduction(proc, reduc_stmt, memory=DRAM, nth_loop=2, unroll=False):
    # Auto-coersion
    if isinstance(unroll, bool):
        unroll = (unroll, unroll)

    reduc_stmt = proc.forward(reduc_stmt)
    reduc_loop = get_enclosing_loop(proc, reduc_stmt, nth_loop)

    # This is technically a duplicate check
    if not is_loop(proc, reduc_loop) and is_single_stmt_loop(proc, reduc_loop):
        raise BLAS_SchedulingError(
            """
        Incorrect structure, must of the form:
            for ...    <---- nth-loop
                for ...
                    ...
                        reduction
                    ...
        """
        )

    # Stage reduction around the loop we are reducing over
    proc = reorder_loops(proc, reduc_loop)
    # This will effectively check that the lhs is actually a reduction over the loop
    proc = auto_stage_mem(
        proc, reduc_stmt, next(get_unique_names(proc)), n_lifts=nth_loop - 1, accum=True
    )
    proc = simplify(proc)

    # Set the memory of the newly created buffer
    reduc_loop = proc.forward(reduc_loop)
    alloc = reduc_loop.prev().prev()
    proc = set_memory(proc, alloc, memory)

    # Parallelize the reduction
    reduc_loop = proc.forward(reduc_loop)
    proc = parallelize_and_lift_alloc(proc, reduc_loop.prev().prev())

    # Fission the zero and store-back stages
    proc = fission(proc, reduc_loop.before())
    proc = fission(proc, reduc_loop.after())

    # Reorder the loop nest back
    reduc_loop = proc.forward(reduc_loop)
    proc = reorder_loops(proc, reduc_loop.parent())

    # Unroll any loops
    reduc_loop = proc.forward(reduc_loop)
    if unroll[0]:
        proc = unroll_loop(proc, reduc_loop.prev())
    if unroll[1]:
        proc = unroll_loop(proc, reduc_loop.next())

    return proc


def interleave_outer_loop_with_inner_loop(
    proc, outer_loop_cursor, inner_loop_cursor, interleave_factor
):
    """
    for i in seq(0, hi):
        B1;
        for j in seq(0, hi'):
            B;
        B2;

    --->

    for io in seq(0, hi / interleave_factor):
        for ii in seq(0, interleave_factor):
            B1;
        for j in seq(0, hi'):
            for ii in seq(0, interleave_factor):
                B;
        for ii in seq(0, interleave_factor):
            B2;

    for io in seq(0, hi % interleave_factor):
        B1;
        for j in seq(0, hi'):
            B;
        B2;
    """
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

    for stmt in middle_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if not isinstance(inner_loop_cursor.prev(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.before())

    inner_loop_cursor = proc.forward(inner_loop_cursor)
    if not isinstance(inner_loop_cursor.next(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.after())

    proc = simplify(proc)
    inner_loop_cursor = proc.forward(inner_loop_cursor)
    proc = reorder_loops(proc, inner_loop_cursor.parent())

    return proc


def vectorize(
    proc,
    loop_cursor,
    vec_width,
    interleave_factor,
    accumulators_count,
    memory_type,
    precision,
    instructions,
    vectorize_tail=True,
):
    # Check pre-conditions
    if not isinstance(loop_cursor, ForCursor):
        raise BLAS_SchedulingError("Expected loop_cursor to be a ForCursor")

    if not isinstance(vec_width, int) and vec_width > 1:
        raise BLAS_SchedulingError("Expected vec_width to be an integer > 1")

    if not isinstance(interleave_factor, int) and interleave_factor > 0:
        raise BLAS_SchedulingError("Expected interleave_factor to be an integer > 1")

    if not isinstance(accumulators_count, int) and accumulators_count > 0:
        raise BLAS_SchedulingError("Expected accumulators_count to be an integer > 1")

    # You add multiple accumulators to increase your ILP.
    # If you don't interleave the execution by at least that
    # much, there is no reason to use mutliple accumulators.
    if interleave_factor % accumulators_count != 0:
        raise BLAS_SchedulingError(
            "Expected interleave_factor % accumulators_count == 0"
        )

    # Forward argument cursors
    loop_cursor = proc.forward(loop_cursor)

    is_perfect = (
        isinstance(loop_cursor.hi(), LiteralCursor)
        and isinstance(loop_cursor.lo(), LiteralCursor)
        and (loop_cursor.hi().value() - loop_cursor.lo().value()) % vec_width == 0
    )
    tail = "cut" if is_perfect or not vectorize_tail else "guard"

    # Divide the loop to expose parallelism
    proc, cursors = auto_divide_loop(proc, loop_cursor, vec_width, tail=tail)
    outer_loop_cursor = cursors.outer_loop_cursor
    inner_loop_cursor = cursors.inner_loop_cursor

    # Parallelize all reductions
    for stmt in lrn_stmts(proc, inner_loop_cursor.body()):
        try:
            proc = parallelize_reduction(proc, stmt, memory_type)
        except (SchedulingError, BLAS_SchedulingError, TypeError):
            pass

    outer_loop_cursor = proc.forward(outer_loop_cursor)
    inner_loop_cursor = outer_loop_cursor.body()[0]

    if tail == "guard":
        # Generate tail loop
        # We manually cut the loop to get the tail loop so that the tail loop
        # automatically uses the parallelized reduction buffer, instead of
        # accumulating into a scalar. This also means that when you vectorize
        # the tail loop (e.g. using mask instructions). You don't need
        # to do two vector reduction, but only one.
        proc = cut_loop(
            proc, outer_loop_cursor, FormattedExprStr("_ - 1", outer_loop_cursor.hi())
        )

        outer_loop_cursor = proc.forward(outer_loop_cursor)
        tail_loop_cursor = outer_loop_cursor.next().body()[0]

        # Now that we have a tail loop, the conditional in the main loop
        # can be removed
        proc = eliminate_dead_code(proc, inner_loop_cursor.body()[0])

        proc = vectorize_to_loops(
            proc, tail_loop_cursor, vec_width, memory_type, precision
        )

    # We can now expand scalar operations to SIMD in the main loop
    proc = vectorize_to_loops(
        proc, inner_loop_cursor, vec_width, memory_type, precision
    )

    if interleave_factor == 1:
        return replace_all(proc, instructions)

    div_factor = accumulators_count if accumulators_count > 1 else interleave_factor

    if accumulators_count > 1:
        proc, cursors = auto_divide_loop(
            proc, outer_loop_cursor, div_factor, tail="cut"
        )
        outer_loop_cursor = cursors.outer_loop_cursor
        inner_loop_cursor = cursors.inner_loop_cursor
        for stmt in filter(
            lambda x: isinstance(x, (AssignCursor, ReduceCursor)),
            lrn_stmts(proc, inner_loop_cursor.body()),
        ):
            try:
                proc = parallelize_reduction(proc, stmt, memory_type, 3, True)
            except (SchedulingError, BLAS_SchedulingError, TypeError):
                continue
        outer_loop_cursor = proc.forward(outer_loop_cursor)
        inner_loop_cursor = outer_loop_cursor.body()[0]
        inner_loop_cursor = proc.forward(inner_loop_cursor)
        proc = interleave_execution(proc, inner_loop_cursor, interleave_factor)

    proc = interleave_execution(
        proc, outer_loop_cursor, interleave_factor // accumulators_count
    )
    proc = replace_all(proc, instructions)

    return proc


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
                proc, inner_loop, loop, tile_size
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
        proc = push_loop_in(proc, cursors.inner_loop_cursor, depth + 1)
        proc = push_loop_in(proc, cursors.tail_loop_cursor, depth + 1)

    return proc


def auto_stage_mem(proc, cursor, new_buff_name, n_lifts=1, accum=False):
    if not isinstance(cursor, (ReadCursor, ReduceCursor, AssignCursor)):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    cursor = proc.forward(cursor)

    lo = []
    hi = []
    loop = get_enclosing_loop(proc, cursor)
    loops = [loop]
    for _ in range(n_lifts - 1):
        loop = get_enclosing_loop(proc, loop)
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
    return stage_mem(proc, loops[-1], window, new_buff_name, accum=accum)


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
