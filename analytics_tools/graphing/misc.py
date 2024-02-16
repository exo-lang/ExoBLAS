def parse_kernel_name(kernel):
    params = kernel.split("_")
    name = params[0][1:]
    if (
        kernel.startswith("sdsdot")
        or kernel.startswith("dsdot")
        or kernel.startswith("s")
    ):
        return (name, "f32", params[1:])
    if kernel.startswith("d"):
        return (name, "f64", params[1:])
    raise NotImplementedError(f"Cannot determine the precision of kernel {kernel}")


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
    tokens = run_name.split("/")
    tokens = tokens[1:]  # Skip the bench name
    tokens = [token.split(":") for token in tokens]
    return {token[0]: token[1] for token in tokens}


__all__ = [
    "parse_kernel_name",
    "get_elem_bytes",
    "b_to_gb",
    "ns_to_s",
    "run_name_to_dict",
]
