from exo import *
from exo.API_cursors import *
from exo.stdlib.analysis import *

from exceptions import *
from higher_order import filter_cursors


def get_children(proc, cursor=InvalidCursor(), lr=True):

    if isinstance(cursor, InvalidCursor):
        cursor = proc
    elif isinstance(cursor, (StmtCursor, ExprCursor)):
        cursor = proc.forward(cursor)

    def expr_children(expr):
        if isinstance(expr, (ReadCursor, WindowExprCursor)):
            yield from expr.idx()
        elif isinstance(expr, UnaryMinusCursor):
            yield expr.arg()
        elif isinstance(expr, BinaryOpCursor):
            yield expr.lhs()
            yield expr.rhs()
        elif isinstance(expr, BuiltInFunctionCursor):
            yield from expr.args()
        elif isinstance(expr, (LiteralCursor, ReadConfigCursor)):
            pass
        else:
            raise BLAS_SchedulingError(
                f"Got an instance of {type(expr)} which is unsupported."
            )

    def stmt_children(stmt):
        if isinstance(stmt, AllocCursor):
            if stmt.is_tensor():
                yield from stmt.shape()
        elif isinstance(stmt, (AssignCursor, ReduceCursor)):
            yield from stmt.idx()
            yield stmt.rhs()
        elif isinstance(stmt, ForCursor):
            yield stmt.lo()
            yield stmt.hi()
            yield from stmt.body()
        elif isinstance(stmt, IfCursor):
            yield stmt.cond()
            yield from stmt.body()
            if not isinstance(stmt.orelse(), InvalidCursor):
                yield from stmt.orelse()
        elif isinstance(stmt, CallCursor):
            yield from stmt.args()
        elif isinstance(stmt, WindowStmtCursor):
            yield stmt.winexpr()
        elif isinstance(stmt, AssignConfigCursor):
            yield stmt.rhs()
        elif isinstance(stmt, PassCursor):
            pass
        else:
            raise BLAS_SchedulingError(
                f"Got an instance of {type(stmt)} which is unsupported."
            )

    def generator():
        if isinstance(cursor, (ExprListCursor, BlockCursor, tuple, list)):
            for c in cursor:
                yield c
        elif isinstance(cursor, Procedure):
            yield from cursor.args()
            yield from cursor.body()
        elif isinstance(cursor, ArgCursor):
            if cursor.is_tensor():
                yield cursor.shape()
        elif isinstance(cursor, ExprCursor):
            yield from expr_children(cursor)
        elif isinstance(cursor, StmtCursor):
            yield from stmt_children(cursor)
        else:
            raise BLAS_SchedulingError(
                f"Got an instance of {type(cursor)} which is unsupported."
            )

    if lr:
        yield from generator()
    else:
        children = list(generator())
        yield from children[::-1]


def get_numeric_children(proc, cursor=InvalidCursor()):
    yield from filter_cursors(is_type_numeric)(proc, get_children(proc, cursor))


def _get_cursors(
    proc, cursor=InvalidCursor(), node_first=False, lr=True, pred=lambda p, c: True
):

    if not isinstance(cursor, (InvalidCursor, ExprListCursor, BlockCursor)):
        cursor = proc.forward(cursor)

    def visist_children(cursor):
        for child in get_children(proc, cursor, lr):
            if pred(proc, child):
                yield from dfs(child)

    def dfs(cursor):
        if node_first:
            yield cursor

        yield from visist_children(cursor)

        if not node_first:
            yield cursor

    if isinstance(cursor, InvalidCursor):
        return visist_children(cursor)
    else:
        return dfs(cursor)


def lrn(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=False)


def nlr(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=True)


def rln(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=False, lr=False)


def nrl(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=True, lr=False)


def lrn_stmts(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=False, pred=is_stmt)


def nlr_stmts(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=True, pred=is_stmt)


def rln_stmts(proc, cursor=InvalidCursor()):
    yield from _get_cursors(
        proc, cursor=cursor, node_first=False, lr=False, pred=is_stmt
    )


def nrl_stmts(proc, cursor=InvalidCursor()):
    yield from _get_cursors(
        proc, cursor=cursor, node_first=True, lr=False, pred=is_stmt
    )


def get_symbols(proc, cursor=InvalidCursor()):
    for c in lrn(proc, cursor):
        if hasattr(c, "name"):
            yield c.name()


def get_declaration(proc, ctxt, name):
    ctxt = proc.forward(ctxt)

    if not isinstance(ctxt, StmtCursor):
        stmt = get_enclosing_stmt(proc, ctxt)
    else:
        stmt = ctxt

    for stmt in get_observed_stmts(stmt):
        if isinstance(stmt, AllocCursor) and stmt.name() == name:
            return stmt
    for arg in proc.args():
        if arg.name() == name:
            return arg
    raise BLAS_SchedulingError("Declaration not found!")


class unique_name:
    cnt = 0

    @staticmethod
    def get():
        name = f"var{unique_name.cnt}"
        unique_name.cnt += 1
        return name


def is_stmt(proc, stmt):
    stmt = proc.forward(stmt)
    return isinstance(stmt, StmtCursor)


def is_if(proc, if_c):
    if_c = proc.forward(if_c)
    return isinstance(if_c, IfCursor)


def is_loop(proc, loop):
    loop = proc.forward(loop)
    return isinstance(loop, ForCursor)


def check_is_loop(proc, loop):
    if not is_loop(proc, loop):
        raise TypeError(f"loop is not a {ForCursor}")


def is_loop_bounds_const(proc, loop):
    check_is_loop(proc, loop)
    loop = proc.forward(loop)
    return isinstance(loop.lo(), LiteralCursor) and isinstance(loop.hi(), LiteralCursor)


def loop_body_len(proc, loop):
    check_is_loop(proc, loop)
    loop = proc.forward(loop)
    return len(loop.body())


def is_single_stmt_loop(proc, loop):
    check_is_loop(proc, loop)
    loop = proc.forward(loop)
    return loop_body_len(proc, loop) == 1


def is_type_numeric(proc, c):
    c = proc.forward(c)
    return hasattr(c, "type") and c.type().is_numeric()


def is_alloc(proc, a):
    a = proc.forward(a)
    return isinstance(a, AllocCursor)


def get_enclosing_scope(proc, cursor, scope_type):
    cursor = proc.forward(cursor)
    cursor = cursor.parent()
    while not isinstance(cursor, (scope_type, InvalidCursor)):
        cursor = cursor.parent()

    if isinstance(cursor, InvalidCursor):
        raise BLAS_SchedulingError("No enclosing scope found")

    return cursor


def get_enclosing_loop(proc, cursor, n=1):
    cursor = proc.forward(cursor)
    for i in range(n):
        cursor = get_enclosing_scope(proc, cursor, ForCursor)
    return cursor


def get_enclosing_if(proc, cursor, n=1):
    cursor = proc.forward(cursor)
    for i in range(n):
        cursor = get_enclosing_scope(proc, cursor, IfCursor)
    return cursor


def get_enclosing_stmt(proc, cursor, n=1):
    cursor = proc.forward(cursor)
    for i in range(n):
        cursor = get_enclosing_scope(proc, cursor, StmtCursor)
    return cursor


def get_my_stmt(proc, cursor):
    cursor = proc.forward(cursor)
    if isinstance(cursor, StmtCursor):
        return cursor
    return get_enclosing_stmt(proc, cursor)


def get_index_in_body(proc, stmt, from_top=True):
    stmt = proc.forward(stmt)
    index = 0
    while not isinstance(stmt.prev(), InvalidCursor):
        stmt = stmt.prev()
        index += 1
    if not from_top:
        parent = get_parent(proc, stmt)
        index = -(len(parent.body()) - index)
    return index


def get_parent(proc, stmt):
    stmt = proc.forward(stmt)
    parent = stmt.parent()
    if isinstance(parent, InvalidCursor):
        parent = proc
    return parent


def get_parents(proc, stmt):
    stmt = proc.forward(stmt)
    stmt = stmt.parent()
    while not isinstance(stmt, InvalidCursor):
        yield stmt
        stmt = stmt.parent()


def get_nth_inner_loop(proc, loop, n):
    loop = proc.forward(loop)
    inner_loops = list(filter_cursors(is_loop)(proc, loop.body()))
    if n >= len(inner_loops):
        raise BLAS_SchedulingError(
            f"Expected exactly at least {n + 1} loops, found {len(inner_loops)}"
        )
    return inner_loops[n]


def get_inner_loop(proc, loop):
    return get_nth_inner_loop(proc, loop, 0)


def is_binop(proc, expr, op=None):
    expr = proc.forward(expr)
    return isinstance(expr, BinaryOpCursor) and (op is None or expr.op() == op)


def is_add(proc, expr):
    return is_binop(proc, expr, "+")


def is_sub(proc, expr):
    return is_binop(proc, expr, "-")


def is_mul(proc, expr):
    return is_binop(proc, expr, "*")


def is_div(proc, expr):
    return is_binop(proc, expr, "/")


def is_mod(proc, expr):
    return is_binop(proc, expr, "%")


def is_builtin(proc, expr, name):
    expr = proc.forward(expr)
    return isinstance(expr, BuiltInFunctionCursor) and expr.name() == name


def is_select(proc, expr):
    return is_builtin(proc, expr, "select")


def is_relu(proc, expr):
    return is_builtin(proc, expr, "relu")


def is_sin(proc, expr):
    return is_builtin(proc, expr, "sin")


def is_literal(proc, expr, value=None):
    expr = proc.forward(expr)
    return isinstance(expr, LiteralCursor) and (value is None or expr.value() == value)


def is_reduce(proc, reduce):
    reduce = proc.forward(reduce)
    return isinstance(reduce, ReduceCursor)


def is_assign(proc, assign):
    assign = proc.forward(assign)
    return isinstance(assign, AssignCursor)


def is_read(proc, read, name=None):
    read = proc.forward(read)
    return isinstance(read, ReadCursor) and (name is None or read.name() == name)


def is_write(proc, write):
    return is_reduce(proc, write) or is_assign(proc, write)


def is_access(proc, access, name=None):
    return (is_read(proc, access) or is_write(proc, access)) and (
        name is None or access.name() == name
    )


def is_window_stmt(proc, window):
    window = proc.forward(window)
    return isinstance(window, WindowStmtCursor)


def is_unary_minus(proc, expr):
    expr = proc.forward(expr)
    return isinstance(expr, UnaryMinusCursor)


def is_call(proc, call, subproc=None):
    call = proc.forward(call)
    return isinstance(call, CallCursor) and (
        subproc is None or call.subproc() == subproc
    )


def is_invalid(proc, inv):
    if isinstance(inv, InvalidCursor):
        return True
    try:
        inv = proc.forward(inv)
        return False
    except InvalidCursorError:
        return True


def is_start_of_body(proc, stmt):
    stmt = proc.forward(stmt)
    return isinstance(stmt.prev(), InvalidCursor)


def is_end_of_body(proc, stmt):
    stmt = proc.forward(stmt)
    return isinstance(stmt.next(), InvalidCursor)


def get_depth(proc, cursor):
    cursor = proc.forward(cursor)

    depth = 1
    while not isinstance(cursor.parent(), InvalidCursor):
        cursor = cursor.parent()
        depth += 1
    return depth


def get_lca(proc, cursor1, cursor2):
    cursor1 = proc.forward(cursor1)
    cursor2 = proc.forward(cursor2)

    depth1 = get_depth(proc, cursor1)
    depth2 = get_depth(proc, cursor2)

    if depth1 < depth2:
        cursor1, cursor2 = cursor2, cursor1
        depth1, depth2 = depth2, depth1

    while depth1 > depth2:
        cursor1 = cursor1.parent()
        depth1 -= 1

    while cursor1 != cursor2:
        cursor1 = cursor1.parent()
        cursor2 = cursor2.parent()

    return cursor1


def get_distance(proc, cursor1, cursor2):
    lca = get_lca(proc, cursor1, cursor2)
    return (
        get_depth(proc, cursor1) + get_depth(proc, cursor2) - 2 * get_depth(proc, lca)
    )


def are_exprs_equal(proc, expr1, expr2):
    expr1 = proc.forward(expr1)
    expr2 = proc.forward(expr2)

    def check(expr1, expr2):
        if type(expr1) != type(expr2):
            return False

        attrs = ["name", "value", "op", "config", "field"]
        for attr in attrs:
            if (
                hasattr(expr1, attr)
                and getattr(expr1, attr)() != getattr(expr2, attr)()
            ):
                return False

        expr1_children = list(get_children(proc, expr1))
        expr2_children = list(get_children(proc, expr2))
        if len(expr1_children) != len(expr2_children):
            return False
        for c1, c2 in zip(expr1_children, expr2_children):
            if not check(c1, c2):
                return False
        return True

    return check(expr1, expr2)


def is_copy(proc, assign):
    assign = proc.forward(assign)
    if not is_assign(proc, assign):
        return False
    if not is_read(proc, assign.rhs()):
        return False
    lhs_decl = get_declaration(proc, assign, assign.name())
    rhs_decl = get_declaration(proc, assign, assign.rhs().name())
    return lhs_decl.mem() is rhs_decl.mem()


def get_bounding_block(proc, cursors):
    cursors = map(lambda c: proc.forward(c), cursors)
    cursors = list(map(lambda c: get_my_stmt(proc, c), cursors))
    depth = list(map(lambda c: get_depth(proc, c), cursors))

    def has_same_parent(cursors):
        parent = cursors[0].parent()
        return all(parent == c.parent() for c in cursors)

    while not has_same_parent(cursors):
        mx = max(depth)
        for i, c in enumerate(cursors):
            if depth[i] == mx:
                cursors[i] = c.parent()
                depth[i] -= 1
    cursors = list(map(lambda c: (c, get_index_in_body(proc, c)), cursors))
    cursors = sorted(cursors, key=lambda k: k[1])
    diff = cursors[-1][1] - cursors[0][1]
    block = cursors[0][0].as_block().expand(0, diff)
    return block


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
