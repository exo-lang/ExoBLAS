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
    return proc


def generate_stride_1_proc(template_proc, precision):
    proc = specialize_precision(template_proc, precision)
    proc = rename(proc, proc.name() + "_stride_1")
    for arg in proc.args():
        if arg.is_tensor():
            proc = proc.add_assertion(
                f"stride({arg.name()}, {len(arg.shape()) - 1}) == 1"
            )
    return proc


def export_exo_proc(globals, proc):
    globals[proc.name()] = proc
    globals.setdefault("__all__", []).append(proc.name())
