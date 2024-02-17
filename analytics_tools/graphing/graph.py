import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

from misc import *

SCRIPT_PATH = Path(__file__)
ROOT_PATH = SCRIPT_PATH.parent.parent.parent.resolve()

GRAPHING_ROOT = SCRIPT_PATH.parent.resolve()
GRAPHS_DIR = GRAPHING_ROOT / "graphs"

BENCHMARK_JSONS_DIR = ROOT_PATH / "benchmark_results"


def kernel_graphs_dir(kernel):
    return GRAPHS_DIR / kernel


def help_msg():
    print("python graph.py <kernel name>!")
    exit(1)


def check_args():
    if len(sys.argv) != 2:
        help_msg()


def init_directories(kernel):
    kernel_graphs_dir(kernel).mkdir(parents=True, exist_ok=True)
    assert BENCHMARK_JSONS_DIR.exists()


def rename_lib(dir_name):
    lib_name = dir_name
    if lib_name == "Intel10_64lp_seq":
        lib_name = "MKL"
    elif lib_name == "exo" or lib_name == "Exo":
        lib_name = "Exo"
    elif lib_name == "All":
        lib_name = "OpenBLAS"
    return lib_name


def get_jsons(kernel):
    jsons = {}
    for libdir in BENCHMARK_JSONS_DIR.iterdir():
        assert libdir.is_dir()

        json_path = libdir / f"{kernel}.json"

        libname = rename_lib(libdir.name)

        if not json_path.exists():
            print(f"Results for {libname}/{libdir.name} not found!")
            continue

        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                print(
                    f"Failed parsing {json_path}. Benchmarking likely got interrupted for {libname}/{libdir.name}"
                )
                continue
            jsons[libname] = data

    return jsons


def get_kernel_class(kernel):
    kernels_specs = importlib.import_module("kernels_specs")
    return getattr(kernels_specs, kernel)


def parse_jsons(kernel, jsons):
    """
    Returns a dictionary of the following form:
    {

        'sub_kernel_name1' : {
            'bench_type1' : {'lib1' : [runs], 'lib2' : [runs] }
            'bench_type2' : ...
            ...
        },
        'sub_kernel_name2' :  { ... }
        ...
    }
    """
    kernel_class = get_kernel_class(kernel)
    parsed_jsons = {}
    for libname, json in jsons.items():
        benchmarks = json["benchmarks"]
        for bench in benchmarks:
            obj = kernel_class(bench)
            parsed_jsons.setdefault(obj.sub_kernel_name, {}).setdefault(
                obj.bench_type, {}
            ).setdefault(libname, []).append(obj)
    return parsed_jsons


def plot_bandwidth_throughput(kernel, data, loads=True):
    plt.clf()

    for libname, runs in data.items():
        sorted_runs = sorted(runs)
        assert len(runs) == len(set(runs))  # No duplicates
        x = [run.get_size_param() for run in sorted_runs]
        if loads:
            y = [run.get_load_gbyte_per_sec() for run in sorted_runs]
        else:
            y = [run.get_store_gbyte_per_sec() for run in sorted_runs]
        plt.plot(x, y, label=libname)
    plt.legend()

    unit = "GBytes / Sec"
    bandwith_type = "loads" if loads else "stores"

    plt.xscale("log")
    plt.ylabel(f"{bandwith_type} {unit}")

    some_point = next(iter(data.values()))[0]

    plt.xlabel(some_point.get_graph_description())
    plt.title(some_point.sub_kernel_name)

    filename = (
        GRAPHS_DIR
        / kernel
        / f"{some_point.sub_kernel_name}_{bandwith_type}_throughput.png"
    )
    plt.savefig(filename)


if __name__ == "__main__":
    check_args()

    kernel = sys.argv[1]

    init_directories(kernel)
    jsons = get_jsons(kernel)
    parsed_jsons = parse_jsons(kernel, jsons)

    for bench_type_dict in parsed_jsons.values():
        for data in bench_type_dict.values():
            plot_bandwidth_throughput(kernel, data, loads=True)
            plot_bandwidth_throughput(kernel, data, loads=False)
