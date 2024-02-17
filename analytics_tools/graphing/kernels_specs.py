from misc import *
from dataclasses import dataclass


class kernel:
    def __init__(self, bench):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_size_param(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_flops(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_input_bytes(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_loaded_bytes(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_stored_bytes(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_graph_description(self):
        raise NotImplementedError(f"{__name__} not implemented")

    def get_real_time(self):
        return self.real_time

    def get_load_gbyte_per_sec(self):
        gb = b_to_gb(self.get_loaded_bytes())
        s = ns_to_s(self.get_real_time())
        return gb / s

    def get_store_gbyte_per_sec(self):
        gb = b_to_gb(self.get_stored_bytes())
        s = ns_to_s(self.get_real_time())
        return gb / s


class level_1(kernel):
    def __init__(self, bench):
        self.real_time = float(bench["real_time"])
        assert bench["name"] == bench["run_name"]

        run_name = bench["run_name"]
        run_dict = run_name_to_dict(run_name)

        self.sub_kernel_name = get_libfree_subkernel_name(run_dict["sub_kernel_name"])
        self.bench_type = int(run_dict["bench_type"])
        assert self.bench_type == BENCH_TYPE.level_1.value
        self.N = int(run_dict["N"])
        self.precision = run_dict["precision"]

    def get_size_param(self):
        return self.N

    def __eq__(self, other):
        return self.N == other.N

    def __lt__(self, other):
        return self.N < other.N

    def __hash__(self):
        return self.N

    def get_graph_description(self):
        return "N"


class level_2(kernel):
    def __init__(self, bench):
        self.precision = precision

        self.real_time = float(bench["real_time"])

        assert bench["name"] == bench["run_name"]
        run_name = bench["run_name"]
        run_dict = run_name_to_dict(run_name)
        self.bench_type = "equal"
        self.N = int(run_dict["N"])
        self.M = self.N

    def get_size_param(self):
        return self.N

    def __eq__(self, other):
        return self.N == other.N

    def __lt__(self, other):
        return self.N < other.N

    def get_graph_description(self):
        return "N"


class axpy(level_1):
    def get_flops(self):
        return self.N * 2

    def get_input_bytes(self):
        return self.N * 2 * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return self.N * 2 * get_elem_bytes(self.precision)

    def get_stored_bytes(self):
        return self.N * get_elem_bytes(self.precision)


class rot(level_1):
    def get_flops(self):
        return self.N * 6

    def get_input_bytes(self):
        return self.N * 2 * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return self.N * 2 * get_elem_bytes(self.precision)

    def get_stored_bytes(self):
        return self.N * 2 * get_elem_bytes(self.precision)


class asum(level_1):
    def get_flops(self):
        return self.N * 2

    def get_input_bytes(self):
        return self.N * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return self.N * get_elem_bytes(self.precision)

    def get_stored_bytes(self):
        return 1 * get_elem_bytes(self.precision)


class ger(level_2):
    def get_flops(self):
        return 2 * self.M * self.N + self.M

    def get_input_bytes(self):
        return (self.M * self.N + self.M + self.N) * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return (self.M * self.N * (1 + 1 / 4) + self.N) * get_elem_bytes(self.precision)

    def get_stored_bytes(self):
        return (self.M * self.N) * get_elem_bytes(self.precision)


class trmv(level_2):
    def get_flops(self):
        return self.N * (self.N + 1)

    def get_input_bytes(self):
        return (self.M * self.N + self.M) * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return ((self.N * (self.N + 1)) / 2 * (1 + 1 / 4) + self.N) * get_elem_bytes(
            self.precision
        )

    def get_stored_bytes(self):
        return ((self.N * (self.N + 1)) / 2 * (1 / 4) + self.N) * get_elem_bytes(
            self.precision
        )


__all__ = ["axpy", "rot"]
