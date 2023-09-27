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


def get_enclosing_scope(cursor, scope_type):
    if not scope_type in (ForSeqCursor, IfCursor):
        raise BLAS_SchedulingError("scope type must be ForSeqCursor or IfCursor")

    cursor = cursor.parent()
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


def stage_expr(proc, expr_cursor, new_name, precision="R", memory=DRAM, n_lifts=1):
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
    proc = lift_alloc(proc, alloc_stmt, n_lifts=n_lifts)
    proc = fission(proc, bind_stmt.after(), n_lifts=n_lifts)
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

    def fission_stmts(proc, body, depth=1):
        for stmt in list(body)[:-1]:
            forwarded_stmt = proc.forward(stmt)
            proc = fission(proc, forwarded_stmt.after(), n_lifts=depth)
            forwarded_stmt = proc.forward(stmt)
            if isinstance(stmt, IfCursor):
                fission_stmts(stmt.body(), depth + 1)
            elif isinstance(stmt, ForSeqCursor):
                raise BLAS_SchedulingError("This is an inner loop vectorizer")

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
        else:
            return [expr]

    is_assign_or_reduce = lambda stmt: isinstance(stmt, (AssignCursor, ReduceCursor))
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    def vectorize_rhs(proc, stmt, depth=1):
        if not is_assign_or_reduce(stmt):
            return proc

        flat_rhs = get_expr_subtree_cursors(stmt.rhs(), stmt, False)

        for expr in flat_rhs:
            proc = stage_expr(proc, expr, f"reg", precision, memory_type, depth)

        return proc

    def vectorize_stmt(proc, stmt, depth=1):
        proc = vectorize_rhs(proc, stmt, depth)
        if isinstance(stmt, ReduceCursor):
            alloc_stmt = get_declaration(proc, stmt, stmt.name())
            if alloc_stmt.mem() != memory_type:

                lhs_reg = "reg"
                proc = stage_mem(
                    proc, stmt, f"{stmt.name()}{expr_to_string(stmt.idx())}", lhs_reg
                )
                forwarded_stmt = proc.forward(stmt)

                alloc_cursor = forwarded_stmt.prev().prev()
                proc = stage_alloc(proc, alloc_cursor)

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

    for stmt in inner_loop_cursor.body():
        proc = vectorize_stmt(proc, stmt)

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
    parllel_factor,
    memory_type,
    precision,
    tail="cut",
):
    """
    for i in seq(0, n):
        x[...] += y[i]

    ----->

    xReg: f32[parllel_factor]
    for ii in seq(0, parllel_factor):
        xReg[ii][...] = 0.0
    for io in seq(0, n / parllel_factor):
        for ii in seq(0, vec_width):
            xReg[ii][..] += y[io * parllel_factor + ii]
    for ii in seq(0, parllel_factor):
        x[...] += xReg[ii]

    Returns: (proc, allocation cursors)
    """
    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")

    if not isinstance(reduction_buffers, list):
        reduction_buffers = [reduction_buffers]

    if parllel_factor is None or parllel_factor == 1:
        return proc, []

    loop_cursor = proc.forward(loop_cursor)

    if not (
        isinstance(loop_cursor.hi(), LiteralCursor)
        and loop_cursor.hi().value() > parllel_factor
    ):
        proc = divide_loop(
            proc,
            loop_cursor,
            parllel_factor,
            (loop_cursor.name() + "o", loop_cursor.name() + "i"),
            tail=tail,
        )
        outer_loop_cursor = proc.forward(loop_cursor)
    else:
        raise BLAS_SchedulingError(
            "Cannot parallelize reduction over a loop with hi smaller than vector width"
        )

    proc = reorder_loops(proc, outer_loop_cursor)
    outer_loop_cursor = proc.forward(outer_loop_cursor)
    allocation_cursors = []
    for buffer in reduction_buffers:
        proc = simplify(stage_mem(proc, outer_loop_cursor, buffer, "reg", accum=True))
        outer_loop_cursor = proc.forward(outer_loop_cursor)
        alloc_cursor = outer_loop_cursor.prev().prev()
        allocation_cursors.append(alloc_cursor)
        proc = set_memory(proc, alloc_cursor, memory_type)
        proc = set_precision(proc, alloc_cursor, precision)

        outer_loop_cursor = proc.forward(outer_loop_cursor)
        proc = stage_alloc(proc, outer_loop_cursor.prev().prev())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = fission(proc, forwarded_outer_loop.before())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = fission(proc, forwarded_outer_loop.after())
        forwarded_outer_loop = proc.forward(outer_loop_cursor)
        proc = reorder_loops(proc, forwarded_outer_loop.parent())

    return proc, tuple(allocation_cursors)


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

    if len(reduction_buffers):
        proc, allocation_cursors = parallelize_reduction(
            proc,
            loop_cursor,
            reduction_buffers,
            vec_width,
            memory_type,
            precision,
            tail="guard",
        )
        loop_cursor = proc.forward(loop_cursor)
        inner_loop = loop_cursor.body()[0]
        proc = cut_loop(proc, loop_cursor, FormattedExprStr("_ - 1", loop_cursor.hi()))
        inner_loop = proc.forward(inner_loop)
        proc = remove_if(proc, inner_loop.body()[0])
    else:
        inner_loop = loop_cursor
    proc = vectorize_to_loops(proc, inner_loop, vec_width, memory_type, precision)
    if len(reduction_buffers) and accumulators_count > 1:
        allocation_cursors = [proc.forward(cur) for cur in allocation_cursors]
        loop_cursor = proc.forward(loop_cursor)
        proc, _ = parallelize_reduction(
            proc,
            loop_cursor,
            [f"{i.name()}[0:{vec_width}]" for i in allocation_cursors],
            accumulators_count,
            memory_type,
            precision,
        )
        loop_cursor = proc.forward(loop_cursor)
        if instructions != None:
            proc = replace_all(proc, instructions)
        if interleave_factor is not None and interleave_factor > 0:
            loop_cursor = proc.forward(loop_cursor)
            inner_loop = loop_cursor.body()[0]
            proc = interleave_execution(proc, inner_loop, accumulators_count)
            if interleave_factor % accumulators_count != 0:
                raise BLAS_SchedulingError(
                    "vectorize interleave_factor must be a multiple of the accumulators_count"
                )
            proc = interleave_execution(
                proc, loop_cursor, interleave_factor // accumulators_count
            )

    elif interleave_factor is not None:
        if instructions != None:
            proc = replace_all(proc, instructions)
        proc = interleave_execution(proc, loop_cursor, interleave_factor)

    return proc


def tile_loops(proc, loop_tile_pairs):

    loop_tile_pairs = [(proc.forward(i[0]), i[1]) for i in loop_tile_pairs]

    inner_loops = []
    for i in range(len(loop_tile_pairs)):
        outer_loop = loop_tile_pairs[i][0]
        tile_size = loop_tile_pairs[i][1]
        proc = divide_loop(
            proc,
            outer_loop,
            tile_size,
            (outer_loop.name() + "o", outer_loop.name() + "i"),
            tail="cut",
        )
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
    return proc


def auto_stage_mem(proc, read_cursor, new_buff_name, n_lifts=1):
    if not isinstance(read_cursor, ReadCursor):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    lo = []
    hi = []
    loop = get_enclosing_loop(read_cursor)
    loops = [loop]
    for _ in range(n_lifts - 1):
        loop = get_enclosing_loop(loop)
        loops.append(loop)

    subst = {}
    for i in range(len(loops) - 1, -1, -1):
        loop = loops[i]
        subst[loop.name()] = f"(({expr_to_string(loop.hi(), subst)})-1)"

    for idx in read_cursor.idx():
        hi.append(expr_to_string(idx, subst))

    for key in subst:
        subst[key] = 0

    for idx in read_cursor.idx():
        lo.append(expr_to_string(idx, subst))

    def ith_idx(i):
        if lo[i] == hi[i]:
            return lo[i]
        else:
            return f"{lo[i]}:(({hi[i]})+1)"

    window = ",".join([ith_idx(i) for i in range(len(read_cursor.idx()))])
    window = f"{read_cursor.name()}[{window}]"
    return stage_mem(proc, loops[-1], window, new_buff_name)
