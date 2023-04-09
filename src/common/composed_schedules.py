from __future__ import annotations

import inspect
import textwrap

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import public_cursors as pc

class BLAS_SchedulingError(Exception):
    pass

def expr_to_string(expr_cursor):
    def expr_list_to_string(expr_list):
        expr_str_list = [expr_to_string(i) for i in expr_list]
        if not expr_str_list:
            return ""
        return "[" + ", ".join(expr_str_list) + "]"
    
    if not isinstance(expr_cursor, pc.ExprCursor):
        raise BLAS_SchedulingError("Cursor must be an ExprCursor")
    if isinstance(expr_cursor, pc.ReadCursor):
        name = str(expr_cursor.name())
        idx_str = expr_list_to_string(expr_cursor.idx())
        return f"({name}{idx_str})"
    elif isinstance(expr_cursor, pc.ReadConfigCursor):
        raise BLAS_SchedulingError("ReadConfigCursor is not supported")
    elif isinstance(expr_cursor, pc.LiteralCursor):
        val_str = str(expr_cursor.value())
        return f"({val_str})"
    elif isinstance(expr_cursor, pc.UnaryMinusCursor):
        arg_str  = expr_to_string(expr_cursor.arg)
        return f"(-{arg_str})"
    elif isinstance(expr_cursor, pc.BinaryOpCursor):
        binop_str = expr_cursor.op()
        lhs_str = expr_to_string(expr_cursor.lhs())
        rhs_str = expr_to_string(expr_cursor.rhs())
        return f"({lhs_str}{binop_str}{rhs_str})"
    elif isinstance(expr_cursor, pc.BuiltInCursor):
        name = expr_cursor.name()
        args_str = expr_list_to_string(expr_cursor.args())
        return f"({name}({args_str[1:-1]}))"
    elif isinstance(expr_cursor, pc.WindowExprCursor):
        raise BLAS_SchedulingError("WindowExprCursor is not supported")
    else:
        assert False, "Undefined Type"    
    
def get_enclosing_scope(cursor, scope_type):
    if not scope_type in (pc.ForSeqCursor, pc.IfCursor):
        raise BLAS_SchedulingError("scope type must be ForSeqCursor or IfCursor")
    
    while not isinstance(cursor, (scope_type, pc.InvalidCursor)):
        cursor = cursor.parent()
    
    if isinstance(cursor, pc.InvalidCursor):
        raise BLAS_SchedulingError("No enclosing scope found")
    
    return cursor

def get_enclosing_loop(cursor):
    return get_enclosing_scope(cursor, pc.ForSeqCursor)

def get_enclosing_if(cursor):
    return get_enclosing_scope(cursor, pc.IfCursor)

def stage_expr(proc, expr_cursor, new_name, cse = False):
    expr_cursor = proc.forward(expr_cursor)
    enclosing_loop = get_enclosing_loop(expr_cursor)
    proc = bind_expr(proc, [expr_cursor], new_name, cse=cse)
    proc = expand_dim(proc, new_name, expr_to_string(enclosing_loop.hi()), str(enclosing_loop.name()))
    proc = lift_alloc(proc, new_name, n_lifts=1)
    proc = fission(proc, proc.find(f"{new_name}[_] = _").after())
    return proc
