from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from introspection import get_statemnts


def specialize_precision(template_proc, precision):
    prefix = "s" if precision == "f32" else "d"
    template_name = template_proc.name()
    template_name = template_name.replace("_template", "")
    specialized_proc = rename(template_proc, "exo_" + prefix + template_name)

    for arg in template_proc.args():
        if arg.type().is_numeric():
            specialized_proc = set_precision(specialized_proc, arg, precision)

    for stmt in get_statemnts(template_proc):
        if isinstance(stmt, AllocCursor):
            specialized_proc = set_precision(specialized_proc, stmt, precision)
    return specialized_proc


def generate_stride_any_proc(template_proc, precision):
    proc = specialize_precision(template_proc, precision)
    proc = rename(proc, proc.name() + "_stride_any")
    proc = stage_scalar_args(proc)
    return proc


def generate_stride_1_proc(template_proc, precision):
    proc = specialize_precision(template_proc, precision)
    proc = rename(proc, proc.name() + "_stride_1")
    for arg in proc.args():
        if arg.is_tensor():
            proc = proc.add_assertion(
                f"stride({arg.name()}, {len(arg.shape()) - 1}) == 1"
            )
    proc = stage_scalar_args(proc)
    return proc


def export_exo_proc(globals, proc):
    globals[proc.name()] = proc
    globals.setdefault("__all__", []).append(proc.name())


def bind_builtins_args(proc, body, precision):
    def expr_visitor(proc, expr):
        if isinstance(expr, BuiltInFunctionCursor):
            for arg in expr.args():
                proc = bind_expr(proc, [arg], "arg")
                alloc_cursor = proc.forward(stmt).prev().prev()
                proc = set_precision(proc, alloc_cursor, precision)
            return proc
        elif isinstance(expr, BinaryOpCursor):
            proc = expr_visitor(proc, expr.lhs())
            return expr_visitor(proc, expr.rhs())
        elif isinstance(expr, UnaryMinusCursor):
            return expr_visitor(expr.arg())
        else:
            return proc

    for stmt in body:
        if isinstance(stmt, ForSeqCursor):
            proc = bind_builtins_args(proc, stmt.body(), precision)
        elif isinstance(stmt, IfCursor):
            proc = bind_builtins_args(proc, stmt.body(), precision)
            if not isinstance(stmt.orelse(), InvalidCursor):
                proc = bind_builtins_args(proc, stmt.orelse(), precision)
        elif isinstance(stmt, ReduceCursor):
            proc = expr_visitor(proc, stmt.rhs())
        elif isinstance(stmt, AssignCursor):
            proc = expr_visitor(proc, stmt.rhs())

    return proc


def stage_scalar_args(proc):
    for arg in proc.args():
        if arg.type().is_numeric() and not arg.is_tensor():
            proc = stage_mem(proc, proc.body(), arg.name(), f"{arg.name()}_")
    return proc
