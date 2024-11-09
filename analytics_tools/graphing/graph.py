import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.font_manager

from misc import *
from kernels_specs import level_1, level_2, level_3

from pylab import rcParams

rc_fonts = {
    "font.family": "serif",
    #'font.serif': 'Linux Libertine',
    "font.serif": [
        "Linux Libertine",
        "Linux Libertine O",
        "Linux Libertine Display O",
        "Linux Libertine Initials O",
        "Linux Libertine Mono O",
    ],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(rc_fonts)
plt.rcParams.update(
    {
        "axes.labelpad": 1,
        #'axes.labelsize': 7,
        "axes.linewidth": 0.5,
        #'grid.linewidth': 0.5,
        #'lines.linewidth': 0.75,
        "xtick.major.pad": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.pad": 0.5,
        "ytick.major.width": 0.5,
    }
)

SCRIPT_PATH = Path(__file__)
ROOT_PATH = SCRIPT_PATH.parent.parent.parent.resolve()

GRAPHING_ROOT = SCRIPT_PATH.parent.resolve()
GRAPHS_DIR = GRAPHING_ROOT / "graphs"
PEAKS_JSON = GRAPHING_ROOT / "peaks.json"

#BENCHMARK_JSONS_DIR = ROOT_PATH / "avx2_benchmark_results_level1"
#BACKEND = "AVX2"

EXOBLAS_NAME = "Exo 2"

def get_peaks_json():
    if PEAKS_JSON.exists():
        with open(PEAKS_JSON, "r") as f:
            peaks = json.load(f)
        return peaks
    else:
        print(f"Could not find peaks jsons at: {PEAKS_JSON}")
        return {}


def kernel_graphs_dir(kernel):
    return GRAPHS_DIR / kernel


def help_msg():
    print("python graph.py <kernel name> <backend name (AVX2 or AVX512)> <benchmark dir>")
    exit(1)


def check_args():
    if len(sys.argv) > 5:
        help_msg()


def parse_args():
    check_args()
    kernel = sys.argv[1]

    global BACKEND
    global BENCHMARK_JSONS_DIR
    BACKEND = sys.argv[2]
    BENCHMARK_JSONS_DIR = Path.cwd() / sys.argv[3]

    return kernel


def init_directories(kernel):
    kernel_graphs_dir(kernel).mkdir(parents=True, exist_ok=True)
    assert BENCHMARK_JSONS_DIR.exists()


def rename_lib(dir_name):
    lib_name = dir_name
    if lib_name == "Intel10_64lp_seq":
        lib_name = "MKL"
    elif lib_name == "exo" or lib_name == "Exo":
        lib_name = EXOBLAS_NAME
    elif lib_name == "All":
        lib_name = "OpenBLAS"
    elif lib_name == "FLAME":
        lib_name = "BLIS"
    return lib_name


def get_jsons(target_kernel):
    jsons = {}
    for libdir in BENCHMARK_JSONS_DIR.iterdir():
        assert libdir.is_dir()
        libname = rename_lib(libdir.name)
        jsons[libname] = {}
        for json_path in libdir.iterdir():
            assert json_path.name.endswith(".json")
            kernel = json_path.name[:-5]

            if target_kernel != "all" and target_kernel != kernel:
                continue

            if not json_path.exists():
                print(f"Results for {libname}/{libdir.name} not found!")
                continue

            with open(json_path) as f:
                try:
                    data = json.load(f)
                except json.decoder.JSONDecodeError:
                    print(f"Failed parsing {json_path}. Benchmarking likely got interrupted for {libname}/{libdir.name}")
                    continue
                jsons[libname][kernel] = data

    return jsons


def get_kernel_class(kernel):
    kernels_specs = importlib.import_module("kernels_specs")
    return getattr(kernels_specs, kernel)


def parse_jsons(jsons):
    """
    Returns a dictionary of the following form:
    {
        'kernel1' = {
            'sub_kernel_name1' : {
            'bench_type1' : {'lib1' : [runs], 'lib2' : [runs] }
            'bench_type2' : ...
            ...
            },
            'sub_kernel_name2' :  { ... }
            ...
        }
        'kernel2' = {
            ...
        }
        ...
    }
    """
    parsed_jsons = {}
    for libname, kernel_dict in jsons.items():
        for kernel, json in kernel_dict.items():
            kernel_class = get_kernel_class(kernel)
            benchmarks = json["benchmarks"]
            for bench in benchmarks:
                obj = kernel_class(bench)
                parsed_jsons.setdefault(kernel, {}).setdefault(obj.sub_kernel_name, {}).setdefault(obj.bench_type, {}).setdefault(
                    libname, []
                ).append(obj)
    return parsed_jsons


def plot_bandwidth_throughput(kernel, data, peaks, loads=True):
    bench_type, data = data
    plt.clf()

    fig, ax = plt.subplots()
    for mem_level in "L1", "L2", "L3", "DRAM":
        key = f"peak_{mem_level}_{'load' if loads else 'store'}_avx"
        if peak := peaks.get(key):
            ax.axhline(y=peak, linewidth=1, color="r", linestyle="dotted", label=key.replace("_", " "))

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
    plt.title(some_point.sub_kernel_name + f" ({BACKEND})")

    sub_kernel_dir = GRAPHS_DIR / kernel / some_point.sub_kernel_name
    sub_kernel_dir.mkdir(parents=True, exist_ok=True)
    filename = sub_kernel_dir / f"{some_point.sub_kernel_name}_{bench_type.name}_{bandwith_type}_throughput.png"
    plt.savefig(filename)


def plot_flops_throughput(kernel, data, peaks):
    bench_type, data = data
    plt.clf()

    some_point = next(iter(data.values()))[0]

    peak_flops_key = "peakflops_sp_avx_fma" if some_point.precision == "f32" else "peakflops_avx_fma"
    if flops := peaks.get(peak_flops_key):
        fig, ax = plt.subplots()
        ax.axhline(y=flops, linewidth=2, color="r", label="Peak Flops")

    for libname, runs in data.items():
        sorted_runs = sorted(runs)
        assert len(runs) == len(set(runs))  # No duplicates
        x = [run.get_size_param() for run in sorted_runs]
        y = [run.get_gflops_per_sec() for run in sorted_runs]
        #if verbose:
        #    print(libname, some_point.sub_kernel_name, bench_type.name, ": ")
        #    for s, flops in zip(x, y):
        #        print(s, flops)
        plt.plot(x, y, label=libname)

    plt.legend()

    unit = "GFlops / Sec"

    plt.xscale("log")
    plt.ylabel(unit)

    plt.xlabel(some_point.get_graph_description())
    plt.title(some_point.sub_kernel_name + f" ({BACKEND})")

    sub_kernel_dir = GRAPHS_DIR / kernel / some_point.sub_kernel_name
    sub_kernel_dir.mkdir(parents=True, exist_ok=True)
    filename = sub_kernel_dir / f"{some_point.sub_kernel_name}_{bench_type.name}_flops_throughput.pdf"
    plt.savefig(filename)


def discretize_and_aggregate(results, ranges, agg_func=np.mean):
    aggregated = {func: {range_: [] for range_ in ranges} for func, _ in results}
    for func, data in results:
        for x, t in data:
            for rng in ranges:
                if rng[0] <= x and (not rng[1] or x < rng[1]):
                    aggregated[func][rng].append(t)

    # Calculate average time for each function and range
    for func in aggregated:
        for range_ in ranges:
            times = aggregated[func][range_]
            aggregated[func][range_] = agg_func(times) if times else 0  # Handle case with no data

    return aggregated


# Convert aggregated data into a format suitable for heatmap creation
def prepare_heatmap_data(aggregated):
    data = []
    functions = list(aggregated.keys())
    ranges = list(next(iter(aggregated.values())).keys())
    for func in functions:
        data.append([aggregated[func][range_] for range_ in ranges])
    array = np.array(data)
    non_zero_columns = ~np.all(array == 0, axis=0)
    array = array[:, non_zero_columns]
    ranges = np.array(ranges)
    ranges = ranges[non_zero_columns]
    functions = [key + " " for key in functions]
    return array, functions, ranges


def to_superscript(n):
    # Dictionary mapping of numbers to their superscript counterparts
    superscript_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
    # Convert the input number to string, map each character to its superscript, and join them back into a string
    return "".join(superscript_map.get(digit, "") for digit in str(n))


def plot_geomean_heatmap(level, bench_type, lib, heatmap_data):
    print(f"{level}, {bench_type}, {lib} ")

    islevel2 = False
    if bench_type.name.find("skinny") != -1:
        plt.figure(figsize=(8 * (3.33 / 6), 3.9 * (3.33 / 6)))
        plt.subplots_adjust(bottom=0.22, left=0.17)
    elif bench_type.name.find("level_2") != -1:
        islevel2 = True

    def aggregate(data):
        data = np.array(data)
        return data.prod() ** (1.0 / len(data))

    powers_list = lambda b, mx: [(b**i, b ** (i + 1)) for i in range(0, mx)]

    p4_ranges = powers_list(4, 11)
    p10_ranges = powers_list(10, 9)

    for p, ranges in ((4, p4_ranges), (10, p10_ranges)):
        if islevel2:
            if p == 4:
                plt.figure(figsize=(8 * (3.33 / 6), 13 * (3.33 / 6)))
                plt.subplots_adjust(left=0.17, bottom=0.05, top=0.95)
            else:
                plt.figure(figsize=(6 * (3.33 / 6), 13 * (3.33 / 6)))
                plt.subplots_adjust(left=0.24, bottom=0.05, top=0.95)
        
        plt.clf()
        agg_heatmap_data = discretize_and_aggregate(heatmap_data, ranges, aggregate)
        data, sub_kernels, ranges = prepare_heatmap_data(agg_heatmap_data)

        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", ["red", "lightgreen", "green"], N=256)

        ax = sns.heatmap(
            data,
            annot=True,
            cbar=False,
            fmt=".2f",
            xticklabels=ranges,
            yticklabels=sub_kernels,
            cmap=cmap,
            vmin=0.8,
            vmax=1.2,
            annot_kws={"fontweight": "bold"},
        )

        # Place the ticks in-between the columns
        tick_positions = np.arange(data.shape[1] + 1)
        tick_labels = [r[0] for r in ranges] + [ranges[-1][1]]
        tick_labels = [f"{p}{to_superscript(round(np.emath.logn(p, l)))}" for l in tick_labels]
        plt.xticks(tick_positions, tick_labels, rotation=0)

        # Remove the ticks from the y-axis
        plt.tick_params(axis="y", which="both", length=0)

        level_name = level.__name__.replace("_", " ").capitalize()
        plt.title(f"Runtime of {lib} / {EXOBLAS_NAME}" + f" ({BACKEND})")
        plt.xlabel("N")

        nrows, ncols = data.shape
        ax.axhline(y=0, color="k", linewidth=2)
        ax.axhline(y=nrows, color="k", linewidth=2)
        ax.axvline(x=0, color="k", linewidth=2)
        ax.axvline(x=ncols, color="k", linewidth=2)

        level_dir = GRAPHS_DIR / "all" / level.__name__
        level_dir.mkdir(parents=True, exist_ok=True)
        filename = level_dir / f"{bench_type.name}_p{p}_disc_gmean_{lib}_x_ExoBLAS"
        #png_path = filename.with_suffix('.png')
        pdf_path = filename.with_suffix('.pdf')

        #plt.savefig(png_path)
        plt.savefig(pdf_path)


def plot_speedup_heatmap(parsed_jsons):

    new_data = {}
    libs = set()
    for kernel_dict in parsed_jsons.values():
        for sub_kernel, bench_type_dict in kernel_dict.items():
            for bench_type, data in bench_type_dict.items():
                new_data.setdefault(bench_type, {})[sub_kernel] = data
                libs |= data.keys()

    libs.remove(EXOBLAS_NAME)
    for bench_type in new_data:
        for lib in libs:
            for level in (level_1, level_2, level_3):
                heatmap_data = []
                for sub_kernel in new_data[bench_type]:
                    if EXOBLAS_NAME not in new_data[bench_type][sub_kernel]:
                        continue
                    if lib not in new_data[bench_type][sub_kernel]:
                        continue
                    if not isinstance(new_data[bench_type][sub_kernel][lib][0], level):
                        continue
                    lib_data = sorted(new_data[bench_type][sub_kernel][lib])
                    exoblas_data = sorted(new_data[bench_type][sub_kernel][EXOBLAS_NAME])

                    lib_data_x = [run.get_size_param() for run in lib_data]
                    exoblas_data_x = [run.get_size_param() for run in exoblas_data]

                    if lib_data_x != exoblas_data_x:
                        continue

                    lib_data_y = [run.get_real_time() for run in lib_data]
                    exoblas_data_y = [run.get_real_time() for run in exoblas_data]

                    geomean = [lib / exoblas for lib, exoblas in zip(lib_data_y, exoblas_data_y)]
                    print(f"Adding {sub_kernel} for bench_type = {bench_type} and lib = {lib}")
                    heatmap_data.append((sub_kernel, list(zip(lib_data_x, geomean))))
                if not heatmap_data:
                    continue

                heatmap_data = sorted(heatmap_data, key=lambda x: (x[0][1:], x[0]))
                plot_geomean_heatmap(level, bench_type, lib, heatmap_data)


def plot_kernel(kernel, parsed_jsons, peaks):
    assert len(parsed_jsons) == 1
    parsed_jsons = next(iter(parsed_jsons.values()))
    for bench_type_dict in parsed_jsons.values():
        for data in bench_type_dict.items():
            plot_bandwidth_throughput(kernel, data, peaks, loads=True)
            plot_bandwidth_throughput(kernel, data, peaks, loads=False)
            plot_flops_throughput(kernel, data, peaks)


def plot_all(parsed_jsons, peaks):
    plot_speedup_heatmap(parsed_jsons)


if __name__ == "__main__":
    check_args()
    kernel = parse_args()

    init_directories(kernel)
    jsons = get_jsons(kernel)
    parsed_jsons = parse_jsons(jsons)

    peaks = get_peaks_json()

    if kernel != "all":
        plot_kernel(kernel, parsed_jsons, peaks)
    else:
        plot_all(parsed_jsons, peaks)


