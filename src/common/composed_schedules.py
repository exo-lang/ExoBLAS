from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from introspection import get_stmt_dependencies, get_declaration


class BLAS_SchedulingError(Exception):
    pass


def expr_to_string(expr_cursor):
    def expr_list_to_string(expr_list):
        expr_str_list = [expr_to_string(i) for i in expr_list]
        if not expr_str_list:
            return ""
        return "[" + ", ".join(expr_str_list) + "]"

    if isinstance(expr_cursor, ExprListCursor):
        return expr_list_to_string(expr_cursor)

    if not isinstance(expr_cursor, ExprCursor):
        raise BLAS_SchedulingError("Cursor must be an ExprCursor")
    if isinstance(expr_cursor, ReadCursor):
        name = str(expr_cursor.name())
        idx_str = expr_list_to_string(expr_cursor.idx())
        return f"({name}{idx_str})"
    elif isinstance(expr_cursor, ReadConfigCursor):
        raise BLAS_SchedulingError("ReadConfigCursor is not supported")
    elif isinstance(expr_cursor, LiteralCursor):
        val_str = str(expr_cursor.value())
        return f"({val_str})"
    elif isinstance(expr_cursor, UnaryMinusCursor):
        arg_str = expr_to_string(expr_cursor.arg)
        return f"(-{arg_str})"
    elif isinstance(expr_cursor, BinaryOpCursor):
        binop_str = expr_cursor.op()
        lhs_str = expr_to_string(expr_cursor.lhs())
        rhs_str = expr_to_string(expr_cursor.rhs())
        return f"({lhs_str}{binop_str}{rhs_str})"
    elif isinstance(expr_cursor, BuiltInCursor):
        name = expr_cursor.name()
        args_str = expr_list_to_string(expr_cursor.args())
        return f"({name}({args_str[1:-1]}))"
    elif isinstance(expr_cursor, WindowExprCursor):
        raise BLAS_SchedulingError("WindowExprCursor is not supported")
    else:
        assert False, "Undefined Type"


def get_enclosing_scope(cursor, scope_type):
    if not scope_type in (ForSeqCursor, IfCursor):
        raise BLAS_SchedulingError("scope type must be ForSeqCursor or IfCursor")

    while not isinstance(cursor, (scope_type, InvalidCursor)):
        cursor = cursor.parent()

    if isinstance(cursor, InvalidCursor):
        raise BLAS_SchedulingError("No enclosing scope found")

    return cursor


def get_enclosing_loop(cursor):
    return get_enclosing_scope(cursor, ForSeqCursor)


def get_enclosing_if(cursor):
    return get_enclosing_scope(cursor, IfCursor)


def get_statement(cursor):

    while not isinstance(cursor, StmtCursor):
        cursor = cursor.parent()

    return cursor


def stage_expr(proc, expr_cursor, new_name, precision="R", memory=DRAM):
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

    expr_cursor = proc.forward(expr_cursor)
    enclosing_loop = get_enclosing_loop(expr_cursor)
    stmt = get_statement(expr_cursor)
    proc = bind_expr(proc, [expr_cursor], new_name)
    stmt = proc.forward(stmt)
    bind_stmt = stmt.prev()
    alloc_stmt = bind_stmt.prev()
    proc = set_precision(proc, alloc_stmt, precision)
    proc = set_memory(proc, alloc_stmt, memory)
    proc = expand_dim(
        proc, alloc_stmt, expr_to_string(enclosing_loop.hi()), enclosing_loop.name()
    )
    proc = lift_alloc(proc, alloc_stmt, n_lifts=1)
    proc = fission(proc, bind_stmt.after())
    return proc


def stage_alloc(proc, alloc_cursor):
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
    enclosing_loop = get_enclosing_loop(alloc_cursor)
    proc = expand_dim(
        proc, alloc_cursor, expr_to_string(enclosing_loop.hi()), enclosing_loop.name()
    )
    proc = lift_alloc(proc, alloc_cursor, n_lifts=1)
    return proc


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

    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError(
            "vectorize_to_loops loop_cursor must be a ForSeqCursor"
        )

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

    inner_loop_stmts = list(inner_loop_cursor.body())

    staged_allocs = []

    for stmt in inner_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = stage_alloc(proc, stmt)
            staged_allocs.append(stmt)

    inner_loop_cursor = proc.forward(inner_loop_cursor)
    inner_loop_stmts = list(inner_loop_cursor.body())

    for stmt in inner_loop_stmts[:-1]:
        forwarded_stmt = proc.forward(stmt)
        proc = fission(proc, forwarded_stmt.after())

    def detect_madd(expr):
        return (
            isinstance(expr, BinaryOpCursor)
            and expr.op() == "*"
            and isinstance(expr.parent(), ReduceCursor)
        )

    def get_expr_subtree_cursors(expr, stmt, alias):
        if not isinstance(expr, (BinaryOpCursor, ReadCursor)):
            raise NotImplementedError(
                "vectorizer is limited to BinOps and Read Expressions"
            )

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
            lhs_alias_with_stmt_lhs = lhs.name() == expr.parent().name()
            rhs_alias_with_stmt_lhs = rhs.name() == expr.parent().name()

        lhs_cursors = get_expr_subtree_cursors(
            lhs, stmt, lhs_alias_with_stmt_lhs or children_alias
        )
        rhs_cursors = get_expr_subtree_cursors(rhs, stmt, rhs_alias_with_stmt_lhs)

        if not detect_madd(expr) and not stmt_lhs_in_mem:
            return lhs_cursors + rhs_cursors + [expr]
        else:
            return lhs_cursors + rhs_cursors

    reg_name_counter = 0

    for stmt in inner_loop_stmts:
        flat_rhs = get_expr_subtree_cursors(stmt.rhs(), stmt, False)

        for expr in flat_rhs:
            proc = stage_expr(
                proc, expr, f"reg{reg_name_counter}", precision, memory_type
            )
            reg_name_counter += 1

        if isinstance(stmt, ReduceCursor):
            alloc_stmt = get_declaration(proc, stmt, stmt.name())
            if alloc_stmt.mem() != memory_type:
                lhs_reg = f"reg{reg_name_counter}"
                reg_name_counter += 1

                proc = stage_mem(
                    proc, stmt, f"{stmt.name()}{expr_to_string(stmt.idx())}]", lhs_reg
                )
                forwarded_stmt = proc.forward(stmt)

                alloc_cursor = forwarded_stmt.prev().prev()
                proc = stage_alloc(proc, alloc_cursor)

                forwarded_stmt = proc.forward(stmt)
                proc = fission(proc, forwarded_stmt.after())
                forwarded_stmt = proc.forward(forwarded_stmt)
                proc = fission(proc, forwarded_stmt.before())

                proc = set_memory(proc, alloc_cursor, memory_type)
                proc = set_precision(proc, alloc_cursor, precision)

    for alloc in staged_allocs:
        proc = set_memory(proc, alloc, memory_type)
        proc = set_precision(proc, alloc, precision)

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
    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")

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
            proc = stage_alloc(proc, stmt)

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


def hoist_stmt(proc, stmt_cursor):
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
    if not isinstance(stmt_cursor, StmtCursor):
        raise BLAS_SchedulingError("Cannot hoist cursor that are not statements")

    if isinstance(stmt_cursor, AllocCursor):
        return lift_alloc(proc, stmt_cursor)

    enclosing_loop = get_enclosing_loop(stmt_cursor)
    if enclosing_loop.name() in get_stmt_dependencies(stmt_cursor):
        raise BLAS_SchedulingError(
            "Cannot hoist cursor to a statement that depends on enclosing loop"
        )

    stmt_cursor = proc.forward(stmt_cursor)
    while not isinstance(stmt_cursor.prev(), InvalidCursor):
        proc = reorder_stmts(proc, stmt_cursor.expand(1, 0))
        stmt_cursor = proc.forward(stmt_cursor)

    proc = fission(proc, stmt_cursor.after())
    stmt_cursor = proc.forward(stmt_cursor)
    proc = remove_loop(proc, stmt_cursor.parent())
    return proc


def apply_to_block(proc, block_cursor, stmt_scheduling_op):
    if not isinstance(block_cursor, BlockCursor):
        raise BLAS_SchedulingError("cannot apply to a non-block cursor")

    for stmt in block_cursor:
        try:
            proc = stmt_scheduling_op(proc, stmt)
        except:
            pass

    return proc


def parallelize_reduction(
    proc,
    loop_cursor,
    reduction_buffers,
    vec_width,
    accumulators_count,
    memory_type,
    precision,
):
    """
    for i in seq(0, n):
        x += y[i]

    ----->

    xReg: f32[accumulators_count][vec_width]
    for im in seq(0, accumulators_count):
        for ii in seq(0, vec_width):
            xReg[im][ii] = 0.0
    for io in seq(0, n / vec_width):
        for im in seq(0, accumulators_count):
            for ii in seq(0, vec_width):
                xReg[im][ii] += y[i]
    for im in seq(0, accumulators_count):
        for ii in seq(0, vec_width):
            x += xReg[im][ii]
    """
    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")

    if not isinstance(reduction_buffers, list):
        reduction_buffers = [reduction_buffers]

    loop_cursor = proc.forward(loop_cursor)
    reg_name = lambda name: f"{name}_parallel_reduction_reg"

    if accumulators_count == 1:
        if not (
            isinstance(loop_cursor.hi(), LiteralCursor)
            and loop_cursor.hi().value() > vec_width
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
            raise BLAS_SchedulingError(
                "Cannot parallelize reduction over a loop with hi smaller than vector width"
            )

        proc = reorder_loops(proc, outer_loop_cursor)
        outer_loop_cursor = proc.forward(outer_loop_cursor)
        for buffer in reduction_buffers:
            proc = simplify(
                stage_mem(proc, outer_loop_cursor, buffer, reg_name(buffer), accum=True)
            )
        outer_loop_cursor = proc.forward(outer_loop_cursor)
        proc = stage_alloc(proc, outer_loop_cursor.prev().prev())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = fission(proc, forwarded_outer_loop.before())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = fission(proc, forwarded_outer_loop.after())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = reorder_loops(proc, forwarded_outer_loop.parent())
    else:
        if not (
            isinstance(loop_cursor.hi(), LiteralCursor)
            and loop_cursor.hi().value() > vec_width * accumulators_count
        ):
            proc = divide_loop(
                proc,
                loop_cursor,
                vec_width * accumulators_count,
                (loop_cursor.name() + "o", loop_cursor.name() + "i"),
                tail="cut",
            )
            outer_loop_cursor = proc.forward(loop_cursor)
            inner_loop_cursor = outer_loop_cursor.body()[0]
            proc = divide_loop(
                proc,
                inner_loop_cursor,
                vec_width,
                (loop_cursor.name() + "m", loop_cursor.name() + "i"),
                perfect=True,
            )
            outer_loop_cursor = proc.forward(loop_cursor)
            middle_loop_cursor = proc.forward(inner_loop_cursor)
            inner_loop_cursor = middle_loop_cursor.body()[0]
        else:
            raise BLAS_SchedulingError(
                "Cannot parallelize reduction over a loop with hi smaller than vector_width * accumulators_count"
            )

        proc = reorder_loops(proc, outer_loop_cursor)
        proc = reorder_loops(proc, outer_loop_cursor)
        for buffer in reduction_buffers:
            proc = simplify(
                stage_mem(proc, outer_loop_cursor, buffer, reg_name(buffer), accum=True)
            )
        outer_loop_cursor = proc.forward(outer_loop_cursor)
        alloc_cursor = outer_loop_cursor.prev().prev()
        proc = stage_alloc(proc, alloc_cursor)
        proc = stage_alloc(proc, alloc_cursor)
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = fission(proc, forwarded_outer_loop.before(), n_lifts=2)
        forwarded_outer_loop = proc.forward(forwarded_outer_loop)
        proc = fission(proc, forwarded_outer_loop.after(), n_lifts=2)
        forwarded_outer_loop = proc.forward(forwarded_outer_loop)
        proc = reorder_loops(proc, forwarded_outer_loop.parent())
        forwarded_outer_loop = proc.forward(forwarded_outer_loop)
        proc = reorder_loops(proc, forwarded_outer_loop.parent())
        forwarded_outer_loop = proc.forward(forwarded_outer_loop)
        proc = unroll_loop(proc, forwarded_outer_loop.prev())
        forwarded_outer_loop = proc.forward(forwarded_outer_loop)
        proc = unroll_loop(proc, forwarded_outer_loop.next())

    for buffer in reduction_buffers:
        proc = set_memory(proc, reg_name(buffer), memory_type)
        proc = set_precision(proc, reg_name(buffer), precision)
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
            proc = stage_alloc(proc, stmt)

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
):
    loop_cursor = proc.forward(loop_cursor)

    reduction_buffers = []
    for stmt in loop_cursor.body():
        if isinstance(stmt, ReduceCursor) and len(stmt.idx()) == 0:
            reduction_buffers.append(stmt.name())

    if accumulators_count != None and len(reduction_buffers) > 0:
        if interleave_factor % accumulators_count != 0:
            raise BLAS_SchedulingError(
                "vectorize interleave_factor must be a multiple of the accumulators_count"
            )
        proc = parallelize_reduction(
            proc,
            loop_cursor,
            reduction_buffers,
            vec_width,
            accumulators_count,
            memory_type,
            precision,
        )
        loop_cursor = proc.forward(loop_cursor)
        middle_loop = loop_cursor.body()[0]
        inner_loop = middle_loop.body()[0]
        proc = vectorize_to_loops(proc, inner_loop, vec_width, memory_type, precision)
        if instructions != None:
            proc = replace_all(proc, instructions)
        proc = interleave_execution(proc, middle_loop, accumulators_count)
        proc = interleave_execution(
            proc, loop_cursor, interleave_factor // accumulators_count
        )
    else:
        proc = vectorize_to_loops(proc, loop_cursor, vec_width, memory_type, precision)
        if instructions != None:
            proc = replace_all(proc, instructions)
        proc = interleave_execution(proc, loop_cursor, interleave_factor)
    return proc
