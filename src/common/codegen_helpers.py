from __future__ import annotations
from pathlib import Path
import json

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from inspection import *
from higher_order import *
import exo_blas_config as C
from perf_features import *
from stdlib import *


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
    globals[proc.name()] = cleanup(proc)
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


def identity_schedule(proc, *args, **kwargs):
    return proc


def get_perf_features(proc):
    return {
        "flops": str(count_flops(proc)),
        "flops_upper_bound": str(count_flops(proc, upper=True)),
        "load_mem_traffic": str(count_load_mem_traffic(proc)),
        "load_mem_traffic_upper_bound": str(count_load_mem_traffic(proc, upper=True)),
        "store_mem_traffic": str(count_store_mem_traffic(proc)),
        "store_mem_traffic_upper_bound": str(count_store_mem_traffic(proc, upper=True)),
    }


def export_perf_features(kernel_name, perf_features):
    REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
    PERF_FEATURES_DIR = REPO_ROOT / "perf_features"
    kernel_json = PERF_FEATURES_DIR / f"{kernel_name}.json"
    PERF_FEATURES_DIR.mkdir(exist_ok=True)

    with open(kernel_json, "w") as f:
        json.dump(perf_features, f, sort_keys=True, indent=4, separators=(",", ": "))


def variants_generator(blas_op, precisions=("f32", "f64")):
    def generate(proc, loop_name, *args, globals=None, **kwargs):
        perf_features = {}
        for precision in precisions:
            proc_variant = specialize_precision(proc, precision)

            proc_variant = stage_scalar_args(proc_variant)

            stride_any = generate_stride_any_proc(proc_variant)
            stride_any = bind_builtins_args(stride_any, stride_any.body(), precision)
            export_exo_proc(globals, stride_any)

            stride_1 = generate_stride_1_proc(proc_variant)
            loop = stride_1.find_loop(loop_name)
            algorithm = get_perf_features(stride_1)

            stride_1 = blas_op(stride_1, loop, precision, C.Machine, *args, **kwargs)
            stride_1 = bind_builtins_args(stride_1, stride_1.body(), precision)
            scheduled = get_perf_features(stride_1)

            perf_features[stride_1.name()] = {
                feature: {
                    "algorithm": algorithm[feature],
                    "scheduled": scheduled[feature],
                }
                for feature in algorithm.keys()
            }

            export_exo_proc(globals, stride_1)
        export_perf_features(proc.name(), perf_features)

    return generate
