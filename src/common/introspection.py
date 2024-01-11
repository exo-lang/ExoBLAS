from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *
from exo.stdlib.analysis import *

from exceptions import *


def _get_stmts(proc, block=InvalidCursor(), node_first=False):
    return_args = False
    if isinstance(block, InvalidCursor):
        return_args = True
        block = proc.body()
        # TODO: forward here once block forwarding works
        # block = proc.forward(block)
    elif isinstance(block, StmtCursor):
        block = proc.forward(block)
        block = block.as_block()
    elif not isinstance(block, BlockCursor):
        raise TypeError(
            f"Got type {type(block)}, expected an instance of {BlockCursor} or {StmtCursor}"
        )

    def yield_args():
        for arg in proc.args():
            yield arg

    if node_first and return_args:
        yield from yield_args()

    def traversal(block):
        for s in block:
            if node_first:
                yield s
            if isinstance(s, IfCursor):
                yield from traversal(s.body())
                if not isinstance(s.orelse(), InvalidCursor):
                    yield from traversal(s.orelse())
            elif isinstance(s, ForCursor):
                yield from traversal(s.body())
            elif isinstance(
                s,
                (
                    AssignCursor,
                    ReduceCursor,
                    AssignConfigCursor,
                    PassCursor,
                    AllocCursor,
                    CallCursor,
                    WindowStmtCursor,
                ),
            ):
                pass
            else:
                raise TypeError(f"Type {type(s)} is not supported.")
            if not node_first:
                yield s

    yield from traversal(block)

    if not node_first and return_args:
        yield from yield_args()


def lrn_stmts(proc, block=InvalidCursor()):
    yield from _get_stmts(proc, block=block, node_first=False)


def nlr_stmts(proc, block=InvalidCursor()):
    yield from _get_stmts(proc, block=block, node_first=True)


def _get_exprs(proc, expr, node_first=False):
    if not isinstance(expr, (ExprCursorPrototype, ExprListCursor)):
        raise TypeError(
            f"Got type {type(expr)}, expected an instance of {ExprCursorPrototype} or {ExprListCursor}"
        )

    if isinstance(expr, ExprCursorPrototype):
        expr = proc.forward(expr)

    def traversal(expr):
        if node_first:
            yield expr
        if isinstance(expr, (ExprListCursor, tuple, list)):
            for e in expr:
                yield from traversal(e)
        elif isinstance(expr, (ReadCursor, WindowExprCursor)):
            yield from traversal(expr.idx())
        elif isinstance(expr, UnaryMinusCursor):
            yield from traversal(expr.arg())
        elif isinstance(expr, BinaryOpCursor):
            yield from traversal(expr.lhs())
            yield from traversal(expr.rhs())
        elif isinstance(expr, BuiltInFunctionCursor):
            yield from traversal(expr.args())
        elif isinstance(expr, (LiteralCursor, ReadConfigCursor)):
            pass
        else:
            raise TypeError(f"Type {type(expr)} is not supported.")
        if not node_first:
            yield expr

    yield from traversal(expr)


def lrn_exprs(proc, expr):
    yield from _get_exprs(proc, expr, node_first=False)


def nlr_exprs(proc, expr):
    yield from _get_exprs(proc, expr, node_first=True)


def _get_cursors(proc, cursor=InvalidCursor(), node_first=False):
    if isinstance(cursor, (ExprCursorPrototype, ExprListCursor)):
        yield from _get_exprs(proc, cursor, node_first)
    else:
        for s in _get_stmts(proc, cursor, node_first):
            if node_first:
                yield s
            if isinstance(s, (ArgCursor, AllocCursor)):
                if s.is_tensor():
                    yield from _get_exprs(proc, s.shape(), node_first)
            elif isinstance(s, (AssignCursor, ReduceCursor)):
                yield from _get_exprs(proc, s.idx(), node_first)
                yield from _get_exprs(proc, s.rhs(), node_first)
            elif isinstance(s, ForCursor):
                yield from _get_exprs(proc, s.lo(), node_first)
                yield from _get_exprs(proc, s.hi(), node_first)
            elif isinstance(s, IfCursor):
                yield from _get_exprs(proc, s.cond())
            elif isinstance(s, CallCursor):
                yield from _get_exprs(proc, s.args(), node_first)
            elif isinstance(s, WindowExprCursor):
                yield from _get_exprs(proc, s.idx(), node_first)
            elif isinstance(s, (ReadConfigCursor, LiteralCursor)):
                pass
            else:
                raise TypeError(f"Type {type(s)} is not supported.")
            if not node_first:
                yield s


def lrn(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=False)


def nlr(proc, cursor=InvalidCursor()):
    yield from _get_cursors(proc, cursor=cursor, node_first=True)


def get_symbols(proc, cursor=InvalidCursor()):
    for c in lrn(proc, cursor):
        if hasattr(c, "name"):
            yield c.name()


def get_declaration(proc, stmt_context, name):
    for stmt in get_observed_stmts(stmt_context):
        if isinstance(stmt, AllocCursor) and stmt.name() == name:
            return stmt
    for arg in proc.args():
        if arg.name() == name:
            return arg
    return None


def get_unique_names(proc):
    cnt = 0
    syms = set(get_symbols(proc))
    while cnt < 100:
        name = f"var{cnt}"
        cnt += 1
        if name in syms:
            continue
        yield name


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


def get_enclosing_scope(proc, cursor, scope_type):
    if not scope_type in (ForCursor, IfCursor):
        raise BLAS_SchedulingError("scope type must be ForCursor or IfCursor")

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
