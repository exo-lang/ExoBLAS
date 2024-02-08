from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from introspection import *
from higher_order import *
import exo_blas_config as C


def specialize_precision(proc, precision, all_buffs=True):
    assert precision in {"f32", "f64"}
    prefix = "s" if precision == "f32" else "d"
    template_name = proc.name()
    proc = rename(proc, prefix + template_name)

    def has_type_R(proc, s, *arg):
        if not isinstance(s, (AllocCursor, ArgCursor)):
            return False
        return s.type() == ExoType.R

    set_R_type = predicate(set_precision, has_type_R)

    def is_numeric(proc, s, *arg):
        if not isinstance(s, (AllocCursor, ArgCursor)):
            return False
        return s.type().is_numeric()

    set_numerics = predicate(set_precision, is_numeric)

    set_type = set_numerics if all_buffs else set_R_type
    proc = apply(set_type)(proc, proc.args(), precision)
    proc = make_pass(set_type)(proc, proc.body(), precision)
    return proc


def generate_stride_any_proc(proc):
    proc = rename(proc, proc.name() + "_stride_any")
    return proc


def generate_stride_1_proc(proc):
    proc = rename(proc, proc.name() + "_stride_1")
    for arg in proc.args():
        if arg.is_tensor():
            proc = proc.add_assertion(
                f"stride({arg.name()}, {len(arg.shape()) - 1}) == 1"
            )
    return proc


def export_exo_proc(globals, proc):
    proc = rename(proc, f"exo_{proc.name()}")
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


def variants_generator(blas_op):
    def generate(proc, loop_name, *args, globals=None):
        for precision in ("f32", "f64"):
            proc_variant = specialize_precision(proc, precision)

            proc_variant = stage_scalar_args(proc_variant)

            stride_any = generate_stride_any_proc(proc_variant)
            stride_any = bind_builtins_args(stride_any, stride_any.body(), precision)
            export_exo_proc(globals, stride_any)

            stride_1 = generate_stride_1_proc(proc_variant)
            loop = stride_1.find_loop(loop_name)
            stride_1 = blas_op(stride_1, loop, precision, C.Machine, *args)
            stride_1 = bind_builtins_args(stride_1, stride_1.body(), precision)
            export_exo_proc(globals, stride_1)

    return generate
