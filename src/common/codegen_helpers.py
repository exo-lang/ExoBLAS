from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import inspect

from exo import *
from exo.libs.memories import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from inspection import *
from higher_order import *
from exo_blas_config import Machine
from perf_features import *
from stdlib import *
from cblas_enums import *
from blaslib import *


def generate_stride_any_proc(proc):
    proc = rename(proc, proc.name() + "_stride_any")
    return proc


def generate_stride_1_proc(proc):
    proc = rename(proc, proc.name() + "_stride_1")
    for arg in proc.args():
        if arg.is_tensor():
            proc = proc.add_assertion(f"stride({arg.name()}, {len(arg.shape()) - 1}) == 1")
    return proc


@dataclass
class paramseter_variants_cursors:
    calls: List

    def __iter__(self):
        yield self.calls


def simplify_triang_loop(proc, block, mapping):
    if "Uplo" not in mapping:
        return proc

    def extract_cut_point(proc, loop):
        loop = proc.forward(loop)
        # Target format (Uplo == CblasUpperValue and iter >= cut_point) or (Uplo == CblasLowerValue and iter < cut_point)
        if len(loop.body()) != 1 or not is_if(proc, loop.body()[0]):
            return None
        cond = loop.body()[0].cond()
        if not is_or(proc, cond):
            return None
        cut_point = cond.lhs().rhs().rhs() if mapping["Uplo"] == CblasUpperValue else cond.rhs().rhs().rhs()
        return expr_to_string(cut_point)

    def rewrite(proc, loop):
        loop = proc.forward(loop)
        cut_point = extract_cut_point(proc, loop)
        if cut_point is not None:
            proc, (loop1, loop2) = cut_loop_(proc, loop, cut_point, rc=True)
            proc = shift_loop(proc, loop2, 0)
            proc = eliminate_dead_code(proc, loop1.body()[0])
            proc = eliminate_dead_code(proc, loop2.body()[0])
            proc = delete_pass(proc)
        return proc

    loops = filter_cursors(is_loop)(proc, lrn(proc, block))
    return apply(attempt(rewrite))(proc, loops)


def generate_parameters_variants(proc, rc=False):
    # Extract Relevant Parameters
    params = []
    for param in proc.args():
        name = param.name()
        if name in Cblas_params_values:
            params.append(name)

    if not params:
        if not rc:
            return proc
        return proc, paramseter_variants_cursors([])

    for param in params:
        assertion = [f"{param} == {v}" for v in Cblas_params_values[param]]
        assertion = " or ".join(assertion)
        proc = proc.add_assertion(assertion)
    # Eliminate any parameter control from within the computation
    for param in params[::-1]:
        value = Cblas_params_values[param][0]
        proc = specialize(proc, proc.body(), f"{param} == {value}")
    proc = dce(proc)
    calls = []

    # Extract the different cases as their own subprocs and schedule them
    def traverse(proc, block, params, suffix, mapping):
        block = proc.forward(block)
        if len(params) == 0:
            name = proc.name() + "_" + suffix
            proc = simplify_triang_loop(proc, block, mapping)
            proc, _ = extract_subproc(proc, block, name)
            call = proc.forward(block)[0]
            calls.append((call, mapping))
            return proc

        if_c = block[0]
        param = params[0]
        param_value = Cblas_params_values[param][0]

        letter = Cblas_suffix[param_value]
        proc = traverse(proc, if_c.body(), params[1:], suffix + letter, mapping | {param: param_value})

        letter = Cblas_suffix[param_value + 1]
        proc = traverse(proc, if_c.orelse(), params[1:], suffix + letter, mapping | {param: param_value + 1})
        return proc

    proc = traverse(proc, proc.body(), params, "", {})
    if not rc:
        return proc
    return simplify(proc), paramseter_variants_cursors(calls)


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


def is_param_optional(func):
    signature = inspect.signature(func)

    def check(name):
        param = signature.parameters.get(name)
        return param and param.default is not inspect.Parameter.empty

    return check


def variants_generator(blas_op, opt_precisions=("f32", "f64"), targets=("avx2", "avx512", "neon")):
    def generate(proc, loop_name, *args, globals=None, **kwargs):
        perf_features = {}
        for precision in ("f32", "f64"):
            proc_variant = blas_specialize_precision(proc, precision)

            stride_any = generate_stride_any_proc(proc_variant)
            stride_any = stage_scalar_args(stride_any)
            stride_any = bind_builtins_args(stride_any, stride_any.body(), precision)
            stride_any = generate_parameters_variants(stride_any)
            export_exo_proc(globals, stride_any)

            stride_1 = generate_stride_1_proc(proc_variant)
            algorithm = get_perf_features(stride_1)

            stride_1, (calls,) = generate_parameters_variants(stride_1, rc=True)
            if precision in opt_precisions and Machine.name in targets:
                if not calls:
                    loop = stride_1.find_loop(loop_name)
                    stride_1 = blas_op(stride_1, loop, precision, Machine, *args, **kwargs)
                else:
                    for (call, mapping) in calls:
                        call = stride_1.forward(call)
                        subproc = call.subproc()
                        loop = subproc.find_loop(loop_name)
                        check = is_param_optional(blas_op)
                        filtered_mapping = dict(filter(lambda p: check(p[0]), mapping.items()))
                        subproc = blas_op(subproc, loop, precision, Machine, *args, **kwargs, **filtered_mapping)
                        stride_1 = call_eqv(stride_1, call, subproc)

            stride_1 = stage_scalar_args(stride_1)
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
