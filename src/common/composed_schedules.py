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
    
    if isinstance(expr_cursor, pc.ExprListCursor):
        return expr_list_to_string(expr_cursor)

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
    proc = expand_dim(proc, new_name, expr_to_string(enclosing_loop.hi()), enclosing_loop.name())
    proc = lift_alloc(proc, new_name, n_lifts=1)
    proc = fission(proc, proc.find(f"{new_name}[_] = _").after())
    return proc

def vectorize(proc, loop_cursor, vec_width, memory_type, precision):
    if not isinstance(loop_cursor, pc.ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")
    
    loop_cursor = proc.forward(loop_cursor)
    
    if not (isinstance(loop_cursor.hi(), pc.LiteralCursor) and loop_cursor.hi().value() == vec_width):
        proc = divide_loop(proc, loop_cursor, vec_width, \
            (loop_cursor.name() + "o", loop_cursor.name() + "i"), tail="cut")
    
        outer_loop_cursor = proc.forward(loop_cursor)
        inner_loop_cursor = outer_loop_cursor.body()[0]
    else:
        inner_loop_cursor = loop_cursor
    
    inner_loop_stmts = list(inner_loop_cursor.body())
    
    for stmt in inner_loop_stmts:
        assert not isinstance(stmt, pc.AllocCursor)

    for stmt in inner_loop_stmts[:-1]:
        proc = fission(proc, stmt.after())
    
    def detect_madd(expr):
        return isinstance(expr, pc.BinaryOpCursor) and \
                expr.op() == "*" and \
                isinstance(expr.parent(), pc.ReduceCursor)
    
    def get_expr_subtree_cursors(expr):
        if not isinstance(expr, pc.BinaryOpCursor):
            return [expr]

        lhs_cursors = get_expr_subtree_cursors(expr.lhs())
        rhs_cursors = get_expr_subtree_cursors(expr.rhs())
        
        if not detect_madd(expr):
            return lhs_cursors + rhs_cursors + [expr]
        else:
            return lhs_cursors + rhs_cursors

    reg_name_counter = 0
    
    for stmt in inner_loop_stmts:
        flat_rhs = get_expr_subtree_cursors(stmt.rhs())

        for expr in flat_rhs:
            proc = stage_expr(proc, expr, f"reg{reg_name_counter}")
            reg_name_counter += 1
        
        if isinstance(stmt, pc.ReduceCursor):
            lhs_reg = f"reg{reg_name_counter}"
            reg_name_counter += 1
            
            proc = stage_mem(proc, stmt, f"{stmt.name()}{expr_to_string(stmt.idx())}]", lhs_reg)
            
            forwarded_stmt = proc.forward(stmt)
            enclosing_loop = get_enclosing_loop(forwarded_stmt)
            proc = expand_dim(proc, lhs_reg, vec_width, enclosing_loop.name())
            proc = lift_alloc(proc, lhs_reg)
            
            forwarded_stmt = proc.forward(forwarded_stmt)
            proc = fission(proc, forwarded_stmt.after())
            forwarded_stmt = proc.forward(forwarded_stmt)
            proc = fission(proc, forwarded_stmt.before())
    
    for i in range(reg_name_counter):
        proc = set_memory(proc, f"reg{i}", memory_type)
        proc = set_precision(proc, f"reg{i}", precision)

    return proc
