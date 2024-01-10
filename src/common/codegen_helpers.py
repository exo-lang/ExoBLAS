from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from introspection import *


def specialize_precision(template_proc, precision):
    prefix = "s" if precision == "f32" else "d"
    template_name = template_proc.name()
    template_name = template_name.replace("_template", "")
    specialized_proc = rename(template_proc, "exo_" + prefix + template_name)

    for arg in template_proc.args():
        if arg.type().is_numeric():
            specialized_proc = set_precision(specialized_proc, arg, precision)

    for stmt in lrn_stmts(template_proc):
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
    globals[proc.name()] = simplify(proc)
    globals.setdefault("__all__", []).append(proc.name())


def bind_builtins_args(proc, block, precision):
    stmt = InvalidCursor()
    for c in nlr(proc, block):
        if isinstance(c, BuiltInFunctionCursor):
            for arg in c.args():
                proc = bind_expr(proc, arg, "arg")
                stmt = proc.forward(stmt)
                alloc = stmt.prev().prev()
                proc = set_precision(proc, alloc, precision)
        elif isinstance(c, StmtCursor):
            stmt = c
    return proc


def stage_scalar_args(proc):
    for arg in proc.args():
        if arg.type().is_numeric() and not arg.is_tensor():
            proc = stage_mem(proc, proc.body(), arg.name(), f"{arg.name()}_")
    return proc
