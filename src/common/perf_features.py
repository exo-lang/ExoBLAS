import sympy as sp
from collections import ChainMap

from exo import *
from exo.API_cursors import *

from inspection import *


def _lift_idx_to_sympy(e, syms, upper):
    if isinstance(e, ReadCursor):
        return syms[e.name()]
    elif isinstance(e, UnaryMinusCursor):
        return -_lift_idx_to_sympy(e.arg(), syms, upper)
    elif isinstance(e, BinaryOpCursor):
        lhs = _lift_idx_to_sympy(e.lhs(), syms, upper)
        rhs = _lift_idx_to_sympy(e.rhs(), syms, upper)
        if e.op() == "+":
            return lhs + rhs
        elif e.op() == "-":
            return lhs - rhs
        elif e.op() == "*":
            return lhs * rhs
        elif e.op() == "/":
            if upper:
                return lhs / rhs
            else:
                return lhs // rhs
        elif e.op() == "%":
            if upper:
                return rhs - 1
            else:
                return lhs % rhs
        else:
            assert False, f"Bad case ({e.op()})"
    elif isinstance(e, LiteralCursor):
        return sp.Integer(e.value())
    else:
        assert False, f"Bad case ({type(e)})"


def get_ops(proc, upper=False):
    def get_numeric_children(proc, cursor=InvalidCursor()):
        check = lambda c: hasattr(c, "type") and c.type().is_numeric()
        yield from filter(check, get_children(proc, cursor))

    zero = sp.Integer(0)
    one = sp.Integer(1)

    def get_expr_ops(expr):
        if isinstance(expr, (BuiltInFunctionCursor, BinaryOpCursor, UnaryMinusCursor)):
            children_ops = sum(
                [get_expr_ops(c) for c in get_numeric_children(proc, expr)]
            )
            return one + children_ops
        else:
            return zero

    integers = not upper

    def get_stmt_ops(cursor, syms):
        if isinstance(cursor, BlockCursor):
            return sum([get_stmt_ops(s, syms) for s in cursor])
        elif isinstance(cursor, InvalidCursor):
            return zero
        elif isinstance(cursor, AssignCursor):
            return get_expr_ops(cursor.rhs())
        elif isinstance(cursor, ReduceCursor):
            return one + get_expr_ops(cursor.rhs())
        elif isinstance(cursor, CallCursor):
            return get_ops(cursor.subproc())
        elif isinstance(cursor, ForCursor):
            lo = _lift_idx_to_sympy(cursor.lo(), syms, upper)
            hi = _lift_idx_to_sympy(cursor.hi(), syms, upper)
            new_syms = syms.new_child()
            iter_sym = sp.Symbol(cursor.name(), integer=integers, positive=integers)
            new_syms[cursor.name()] = iter_sym
            stmts_ops = [get_stmt_ops(s, new_syms) for s in cursor.body()]
            sums = [
                sp.summation(s_ops, (iter_sym, lo, hi - one)) for s_ops in stmts_ops
            ]
            return sum(sums)
        else:
            return zero

    syms = ChainMap()
    for arg in proc.args():
        if arg.type().is_indexable():
            is_size = arg.type() == ExoType.Size
            syms[arg.name()] = sp.Symbol(
                arg.name(), integer=integers, positive=is_size and integers
            )

    ops = get_stmt_ops(proc.body(), syms)
    ops = sp.simplify(ops)
    return ops
