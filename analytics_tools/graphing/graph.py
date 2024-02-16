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
    kernel_name, _, _ = parse_kernel_name(kernel)
    kernel_graphs_dir(kernel_name).mkdir(parents=True, exist_ok=True)
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
    name, _, _ = parse_kernel_name(kernel)
    kernels_specs = importlib.import_module("kernels_specs")
    return getattr(kernels_specs, name)


def parse_jsons(kernel, jsons):
    """
    Returns a dictionary of the following form:
    {
        'bench_type1' : {'lib1' : [runs], 'lib2' : [runs] }
        'bench_type2' : ...
        ...
    }
    """
    kernel_class = get_kernel_class(kernel)
    parsed_jsons = {}
    for libname, json in jsons.items():
        benchmarks = json["benchmarks"]
        for bench in benchmarks:
            obj = kernel_class(kernel, bench)
            parsed_jsons.setdefault(obj.type, {}).setdefault(libname, []).append(obj)
    return parsed_jsons


def plot_bandwidth_throughput(kernel, parsed_jsons, loads=True):
    kernel_name, precision, params = parse_kernel_name(kernel)

    def plot(data):
        plt.clf()

        for libname, runs in data.items():
            sorted_runs = sorted(runs)
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
        plt.xlabel(next(iter(data.values()))[0].get_graph_description())
        plt.title(kernel)

        filename = GRAPHS_DIR / kernel_name / f"{kernel}_{bandwith_type}_throughput.png"
        plt.savefig(filename)

    for data in parsed_jsons.values():
        plot(data)


if __name__ == "__main__":
    check_args()

    kernel = sys.argv[1]

    init_directories(kernel)
    jsons = get_jsons(kernel)
    parsed_jsons = parse_jsons(kernel, jsons)

    plot_bandwidth_throughput(kernel, parsed_jsons)
    plot_bandwidth_throughput(kernel, parsed_jsons, loads=False)
