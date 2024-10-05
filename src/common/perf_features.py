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


def _count(proc, count_assign, count_reduce, upper=False):
    zero = sp.Integer(0)
    one = sp.Integer(1)

    integers = not upper

    def get_stmt_ops(cursor, syms):
        if isinstance(cursor, BlockCursor):
            return sum(get_stmt_ops(s, syms) for s in cursor)
        elif isinstance(cursor, InvalidCursor):
            return zero
        elif isinstance(cursor, AssignCursor):
            return count_assign(proc, cursor)
        elif isinstance(cursor, ReduceCursor):
            return count_reduce(proc, cursor)
        elif isinstance(cursor, CallCursor):
            return _count(cursor.subproc(), count_assign, count_reduce, upper)
        elif isinstance(cursor, ForCursor):
            lo = _lift_idx_to_sympy(cursor.lo(), syms, upper)
            hi = _lift_idx_to_sympy(cursor.hi(), syms, upper)
            new_syms = syms.new_child()
            iter_sym = sp.Symbol(cursor.name(), integer=integers, positive=integers)
            new_syms[cursor.name()] = iter_sym
            stmts_ops = [get_stmt_ops(s, new_syms) for s in cursor.body()]
            sums = [sp.summation(s_ops, (iter_sym, lo, hi - one)) for s_ops in stmts_ops]
            return sum(sums)
        else:
            return zero

    syms = ChainMap()
    for arg in proc.args():
        if arg.type().is_indexable():
            is_size = arg.type() == ExoType.Size
            syms[arg.name()] = sp.Symbol(arg.name(), integer=integers, positive=is_size and integers)
    try:
        ops = get_stmt_ops(proc.body(), syms)
        ops = sp.simplify(ops)
        ops = sp.factor(ops)
        return ops
    except:
        return zero


def count_flops(proc, upper=False):
    zero = sp.Integer(0)
    one = sp.Integer(1)

    def get_expr_ops(proc, expr):
        expr = proc.forward(expr)
        if isinstance(expr, (ExternFunctionCursor, BinaryOpCursor, UnaryMinusCursor)):
            children_ops = sum(get_expr_ops(proc, c) for c in get_numeric_children(proc, expr))
            return one + children_ops
        return zero

    count_assign = lambda p, a: get_expr_ops(p, a.rhs())
    count_reduce = lambda p, r: one + get_expr_ops(p, r.rhs())
    return _count(proc, count_assign, count_reduce, upper)


def _traffic_from_DRAM(proc, c):
    zero = sp.Integer(0)
    one = sp.Integer(1)
    c = proc.forward(c)
    decl = get_declaration(proc, c, c.name())
    mem = decl.mem()
    p_bytes = _get_precision_bytes(decl.type())
    return one if issubclass(mem, DRAM) else zero, p_bytes


def _get_precision_bytes(precision):
    d = {
        ExoType.F16: 2,
        ExoType.F32: 4,
        ExoType.F64: 8,
        ExoType.UI8: 1,
        ExoType.I8: 1,
        ExoType.UI16: 2,
        ExoType.I32: 4,
    }
    if precision not in d:
        raise ValueError(f"Unsupported precision {precision}")
    return d[precision]


def _count_mem_traffic(proc, loads, upper=False):
    zero = sp.Integer(0)

    p_bytes_set = set()

    def get_bytes_traffic(p, c):
        num, p_bytes = _traffic_from_DRAM(p, c)
        p_bytes_set.add(p_bytes)
        return num * p_bytes

    def get_expr_loads(proc, expr):
        expr = proc.forward(expr)
        if isinstance(expr, (ExternFunctionCursor, BinaryOpCursor, UnaryMinusCursor)):
            return sum(get_expr_loads(proc, c) for c in get_numeric_children(proc, expr))
        elif isinstance(expr, ReadCursor):
            return get_bytes_traffic(proc, expr)
        return zero

    if loads:
        count_assign = lambda p, a: get_expr_loads(p, a.rhs())
        count_reduce = lambda p, r: get_bytes_traffic(p, r) + get_expr_loads(p, r.rhs())
        traffic = _count(proc, count_assign, count_reduce, upper)
    else:
        count_lhs = lambda p, c: get_bytes_traffic(p, c)
        traffic = _count(proc, count_lhs, count_lhs, upper)

    if len(p_bytes_set) == 1:
        p_bytes = list(p_bytes_set)[0]
        traffic = traffic / p_bytes
        traffic = sp.simplify(traffic)
        return sp.Mul(p_bytes, traffic, evaluate=False)
    else:
        return traffic


def count_load_mem_traffic(proc, upper=False):
    return _count_mem_traffic(proc, True, upper)


def count_store_mem_traffic(proc, upper=False):
    return _count_mem_traffic(proc, False, upper)
