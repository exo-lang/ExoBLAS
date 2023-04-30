from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *


def get_statemnts(proc):
    def get_statemnts_helper(body):
        for stmt in body:
            yield stmt
            if isinstance(stmt, IfCursor):
                yield from get_statemnts_helper(stmt.body())
                yield from get_statemnts_helper(stmt.orelse())
            elif isinstance(stmt, ForSeqCursor):
                yield from get_statemnts_helper(stmt.body())

    return get_statemnts_helper(proc.body())


def get_expr_dependencies(expr):
    if isinstance(expr, ExprListCursor):
        for e in expr:
            yield from get_expr_dependencies(e)
    elif isinstance(expr, ReadCursor):
        yield expr.name()
    elif isinstance(expr, UnaryMinusCursor):
        yield from get_expr_dependencies(expr.arg())
    elif isinstance(expr, BuiltInFunctionCursor):
        yield from get_expr_dependencies(expr.args())
    elif isinstance(expr, BinaryOpCursor):
        yield from get_expr_dependencies(expr.lhs(), expr.rhs())
    elif isinstance(expr, WindowExprCursor):
        yield from get_expr_dependencies(expr.idx())


def get_stmt_dependencies(stmt):
    if isinstance(stmt, BlockCursor):
        for s in stmt:
            yield from get_stmt_dependencies(s)
    elif isinstance(stmt, (ReadCursor, ReduceCursor)):
        yield stmt.name()
        yield from get_expr_dependencies(stmt.idx())
        yield from get_expr_dependencies(stmt.rhs())
    elif isinstance(stmt, CallCursor):
        yield from get_expr_dependencies(stmt.args())
    elif isinstance(stmt, IfCursor):
        yield from get_expr_dependencies(stmt.cond())
        yield from get_stmt_dependencies(stmt.body())
    elif isinstance(stmt, ForSeqCursor):
        yield from get_expr_dependencies(stmt.hi())
        yield from get_stmt_dependencies(stmt.body())
    elif isinstance(stmt, AllocCursor):
        yield from get_expr_dependencies(stmt.shape())
    elif isinstance(stmt, WindowStmtCursor):
        yield from get_expr_dependencies(stmt.winexpr())
