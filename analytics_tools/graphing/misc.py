from enum import Enum


def get_libfree_subkernel_name(sub_kernel_name):
    sub_kernel_name = sub_kernel_name.split("_")
    return "_".join(sub_kernel_name[1:])


def get_elem_bytes(precision):
    elem_bytes = {"f32": 8, "f64": 4}
    if precision not in elem_bytes:
        raise NotImplementedError(f"precision {precision} is not supported")
    return elem_bytes[precision]


def b_to_gb(b):
    return b / ((2**10) ** 3)


def ns_to_s(ns):
    return ns / (10**9)


def run_name_to_dict(run_name):
    run_name = "sub_kernel_name:" + run_name
    tokens = run_name.split("/")
    tokens = [token.split(":") for token in tokens]
    tokens = {token[0]: token[1] for token in tokens}
    tokens["precision"] = f"f{tokens['precision']}"
    return tokens


class BENCH_TYPE(Enum):
    level_1 = 0
    level_2_eq = 1
    level_2_sq = 2
    level_3_eq = 3


level_2_bench_types = {BENCH_TYPE.level_2_eq.value, BENCH_TYPE.level_2_sq.value}
level_3_bench_types = {BENCH_TYPE.level_3_eq.value}

# From netlib `cblas.h`
class CBLAS_TRANSPOSE(Enum):
    CblasNoTrans = (111,)
    CblasTrans = (112,)
    CblasConjTrans = 113


__all__ = [
    "get_libfree_subkernel_name",
    "get_elem_bytes",
    "b_to_gb",
    "ns_to_s",
    "run_name_to_dict",
    "BENCH_TYPE",
    "level_2_bench_types",
    "level_3_bench_types",
    "CBLAS_TRANSPOSE",
]
