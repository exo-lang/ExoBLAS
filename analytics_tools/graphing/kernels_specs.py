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

    def get_input_bytes(self):
        return self.N * self.num_vec * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return self.get_input_bytes()


class level_1_single_vec(level_1):
    def __init__(self, bench):
        self.num_vec = 1
        super().__init__(bench)


class level_1_double_vec(level_1):
    def __init__(self, bench):
        self.num_vec = 2
        super().__init__(bench)


class axpy(level_1_double_vec):
    def get_flops(self):
        return self.N * 2

    def get_stored_bytes(self):
        return self.get_input_bytes() // 2


class rot(level_1_double_vec):
    def get_flops(self):
        return self.N * 6

    def get_stored_bytes(self):
        return self.get_input_bytes()


class rotm(level_1_double_vec):
    def __init__(self, bench):
        super().__init__(bench)

        run_name = bench["run_name"]
        run_dict = run_name_to_dict(run_name)

        self.FLAG = int(run_dict["Flag"])
        self.sub_kernel_name += f", Flag = {self.FLAG}"

    def get_flops(self):
        if self.FLAG == -2:
            return 1
        return self.N * 6

    def get_loaded_bytes(self):
        if self.FLAG == -2:
            return 1
        return self.get_input_bytes()

    def get_stored_bytes(self):
        if self.FLAG == -2:
            return 1
        return self.get_input_bytes()


class asum(level_1_single_vec):
    def get_flops(self):
        return self.N * 2

    def get_stored_bytes(self):
        return 1 * get_elem_bytes(self.precision)


class copy(level_1_double_vec):
    def get_flops(self):
        return 1

    def get_stored_bytes(self):
        return self.get_input_bytes()


class dot(level_1_double_vec):
    def get_flops(self):
        return self.N * 2

    def get_stored_bytes(self):
        return 1 * get_elem_bytes(self.precision)


class dsdot(level_1_double_vec):
    def get_flops(self):
        return self.N * 2

    def get_stored_bytes(self):
        return 1 * get_elem_bytes(self.precision)


class sdsdot(dsdot):
    pass


class scal(level_1_single_vec):
    def get_flops(self):
        return self.N

    def get_stored_bytes(self):
        return self.get_input_bytes()


class swap(level_1_double_vec):
    def get_flops(self):
        return 1

    def get_stored_bytes(self):
        return self.get_input_bytes()


class level_2(kernel):
    def __init__(self, bench):
        self.real_time = float(bench["real_time"])
        assert bench["name"] == bench["run_name"]

        run_name = bench["run_name"]
        run_dict = run_name_to_dict(run_name)

        self.sub_kernel_name = get_libfree_subkernel_name(run_dict["sub_kernel_name"])
        self.bench_type = int(run_dict["bench_type"])
        assert self.bench_type in level_2_bench_types
        self.N = int(run_dict["N"])
        if self.bench_type != BENCH_TYPE.level_2_sq:
            self.M = int(run_dict["M"])
        self.precision = run_dict["precision"]
        if "TransA" in run_dict:
            self.TransA = int(run_dict["TransA"])

    def get_size_param(self):
        return self.N

    def get_cmp_tuple_(self):
        return (self.N, self.M)

    def __eq__(self, other):
        return self.get_cmp_tuple_() == other.get_cmp_tuple_()

    def __lt__(self, other):
        return self.get_cmp_tuple_() < other.get_cmp_tuple_()

    def __hash__(self):
        return hash(self.get_cmp_tuple_())

    def get_graph_description(self):
        if self.bench_type == BENCH_TYPE.level_2_sq.value:
            return "N"
        elif self.bench_type == BENCH_TYPE.level_2_eq.value:
            return "M = N"


class gemv(level_2):
    def get_flops(self):
        return 2 * self.M * self.N + self.M

    def get_input_bytes(self):
        return (self.M * self.N + self.M + self.N) * get_elem_bytes(self.precision)

    def get_loaded_bytes(self):
        return (self.M * self.N + self.M * self.N / 4 + self.M) * get_elem_bytes(
            self.precision
        )

    def get_stored_bytes(self):
        if self.TransA == CBLAS_TRANSPOSE.CblasNoTrans:
            return self.M * get_elem_bytes(self.precision)
        else:
            return ((self.M * self.N) // 4) * get_elem_bytes(self.precision)


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
