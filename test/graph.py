import json
import sys
import matplotlib.pyplot as plt
import math
import os

arith_intensity = {
    "nrm2": 0.5,
    "axpy": 1,
    "dot": 1,
    "asum": 1.25,
    "ger": 3,
    "syr": 3,
    "syr2": 3,
    "trmv": 2,
    "gemv": 2,
    "gbmv": 2,
    "tbmv": 2,
    "sdsdot": 1,
    "dsdot": 1,
    "copy": 0,
    "swap": 0,
    "scal": 1,
    "rot": 3,
    "rotm": 3,
}

read_bound_kernels = {
    "nrm2",
    "axpy",
    "dot",
    "asum",
    "trmv",
    "gemv",
    "gbmv",
    "tbmv",
    "sdsdot",
    "dsdot",
}
write_bound_kernels = {"copy", "swap", "scal", "rot", "rotm", "ger", "syr", "syr2"}
level3_kernels = {"gemm", "symm", "syrk", "syr2k", "trmm", "trsm"}


def mem_footprint(kernel_name, size, wordsize, **kwargs):
    if kernel_name in [
        "nrm2",
        "scal",
        "copy",
        "rot",
        "swap",
        "asum",
        "dot",
        "axpy",
        "sdsdot",
        "dsdot",
        "rotm",
    ]:
        return size * wordsize
    elif kernel_name in ["gemv", "ger", "syr", "syr2", "trmv", "tbmv", "gemm", "syrk"]:
        return size * size * wordsize
    elif kernel_name == "gbmv":
        kl = int(kwargs.get("kl_setting"))
        if kl == 1:
            kl = size / 2
        elif kl == 2:
            kl = size / 4
        else:
            raise
        return (size - kl / 2) * (2 * kl + 1) * wordsize + size * 2
    else:
        raise NotImplementedError(f"Input size of {kernel_name} is not implemented")


def mem_ops(kernel_name, size, wordsize, **kwargs):
    """
    Returns total memory usage of `kernel_name` of dimension `size` in bytes.
    """
    mem_ops = {
        1: {
            "nrm2": 1,
            "scal": 1,
            "copy": 1,
            "rot": 2,
            "swap": 2,
            "asum": 1,
            "dot": 2,
            "axpy": 2,
            "rotm": 2,
            "sdsdot": 2,
            "dsdot": 2,
        },
        2: {"gemv": 1, "ger": 1, "syr": 1, "syr2": 1, "trmv": 0.5, "tbmv": 0.5},
        3: {},
    }

    if kernel_name in mem_ops[1].keys():
        return wordsize * size * mem_ops[1].get(kernel_name, 2)
    elif kernel_name in mem_ops[2].keys():
        # TODO: generalize beyond nxn matrices
        return wordsize * size * size * mem_ops[2].get(kernel_name, 1)
    elif kernel_name in mem_ops[3].keys():
        return wordsize * size * size * mem_ops[3].get(kernel_name, 1)
    elif kernel_name == "gbmv":
        kl = int(kwargs.get("kl_setting"))
        if kl == 1:
            kl = size / 2
        elif kl == 2:
            kl = size / 4
        else:
            raise
        return (size - kl / 2) * (2 * kl + 1) * wordsize + size * 2
    else:
        raise NotImplementedError(f"Memory usage of {kernel_name} is not implemented")


if len(sys.argv) != 4:
    print("python graph.py <kernel name> <benchmark results dir> <graphs_dir>!")
    exit(1)

kernel_name = sys.argv[1]
if kernel_name[0] == "s" or kernel_name == "sdsdot" or kernel_name == "dsdot":
    wordsize = 4
elif kernel_name[0] == "d":
    wordsize = 8
else:
    raise NotImplementedError(f"kernel: {kernel_name} not supported")

benchmark_results_dir = sys.argv[2]
graphs_dir = sys.argv[3]

if not os.path.exists(graphs_dir):
    os.mkdir(graphs_dir)

kernel_graphs_dir = graphs_dir + "/" + kernel_name
if not os.path.exists(kernel_graphs_dir):
    os.mkdir(kernel_graphs_dir)

jsons_dict = {}
for dir in os.listdir(benchmark_results_dir):
    name = dir
    if name == "Intel10_64lp_seq":
        name = "MKL"
    elif name == "exo":
        name = "Exo"
    elif name == "All":
        name = "OpenBLAS"
    json_path = f"{benchmark_results_dir}/{dir}/{kernel_name}.json"
    if not os.path.exists(json_path):
        continue

    with open(json_path) as f:
        data = json.load(f)
    jsons_dict[name] = data


def log_2(x):
    return math.log(x, 2)


def plot_cache_boundries(ymax):
    plt.vlines(x=log_2(192 * 1024), ymin=0, ymax=ymax, color="red", label="L1")
    plt.vlines(
        x=log_2(3 * 1024 * 1024 // 2), ymin=0, ymax=ymax, color="gray", label="L2"
    )
    plt.vlines(x=log_2(9 * 1024 * 1024), ymin=0, ymax=ymax, color="yellow", label="L3")


"""
A dict of raw data
perf_res = {
    (param0, param1, ...) : {bm_name: [(size, time)]}
}
"""
perf_res = {}

# Parse data
for benchmark_name in jsons_dict:
    data = jsons_dict[benchmark_name]
    for d in data["benchmarks"]:
        lis = d["name"].split("/")
        if "Fixture" in lis[0]:
            lis = lis[1:]
        name = benchmark_name
        size = int(lis[1].split(":")[1])  # Words
        params = tuple(lis[2:])

        if kernel_name[1:] in level3_kernels:
            size = []
            for li in lis[1:]:
                n = int(li.split(":")[1])
                size.append(n)
            params = ()

        time = d["cpu_time"]  # ns
        # size : word already
        # time is nanoseconds

        params_dict = perf_res.setdefault(params, {})
        points = params_dict.setdefault(name, [])

        points.append((size, time))

# Sort Raw Data
for params in perf_res:
    params_dict = perf_res[params]
    for name in params_dict:
        params_dict[name] = sorted(params_dict[name])


def peak_bandwidth_plot(params, names_to_points):
    k_name = (
        kernel_name
        if (kernel_name == "sdsdot" or kernel_name == "dsdot")
        else kernel_name[1:]
    )
    bandwidth = False if arith_intensity[k_name] != 0 else True
    scale = lambda x: x * arith_intensity[k_name] if not bandwidth else x

    def get_gbyte_sec(size, time, **kwargs):
        return (mem_ops(k_name, size, wordsize, **kwargs) * 10**9) / (time * 2**30)

    plt.clf()
    for name in names_to_points:
        points = names_to_points[name]

        args = {f: s for f, s in [ele.split(":") for ele in params]}
        x = [log_2(mem_footprint(k_name, p[0], wordsize, **args)) for p in points]
        y = [scale(get_gbyte_sec(p[0], p[1], **args)) for p in points]

        plt.plot(x, y, label=name)

    peak_x = [
        0,
        log_2(32 * 1024),
        log_2(256 * 1024),
        log_2(6 * 1024 * 1024),
        log_2(66 * 1024 * 1024),
    ]
    if k_name in read_bound_kernels:
        peak_y = [
            248890.88 / 1024,
            108097.024 / 1024,
            58041.4464 / 1024,
            22081.9 / 1024,
            22081.9 / 1024,
        ]
    elif k_name in write_bound_kernels:
        peak_y = [
            121653.2 / 1024,
            72869.1 / 1024,
            40314.6 / 1024,
            35841.1264 / 1024,
            35841.1264 / 1024,
        ]
    else:
        assert False, f"unsupported kernel: {kernel_name}"

    peak_y = [scale(y) for y in peak_y]

    peak_x[-1] = x[-1]
    peak_x[0] = x[0]
    plt.plot(peak_x, peak_y, drawstyle="steps-post", label="peak bandwidth")

    if not bandwidth:
        peak_compute_x = [x[0], x[-1]]
        peak_c = (
            128.0 / 2
            if wordsize == 8 or (kernel_name == "sdsdot" or kernel_name == "dsdot")
            else 128.0
        )
        peak_compute_y = [peak_c, peak_c]
        plt.plot(peak_compute_x, peak_compute_y, label="peak compute")

    plt.legend()

    plt.title(f"""{kernel_name}, params: {params}""")
    if not bandwidth:
        plt.ylabel("GFLOPs/sec")
    else:
        plt.ylabel("Gbytes/sec")
    plt.xlabel("log2(bytes)")
    plt.xticks(range(int(x[0]), int(x[-1]), 2))
    fig_file_name = f"{kernel_graphs_dir}/peak_{kernel_name}_{params}.png"
    fig_file_name = fig_file_name.replace(":", "=")  # For Windows compatibility
    plt.savefig(fig_file_name)


def peak_compute_plot(params, names_to_points):
    plt.clf()

    for name in names_to_points:
        points = names_to_points[name]
        x = [
            log_2(
                (p[0][0] * p[0][1] + p[0][0] * p[0][2] + p[0][1] * p[0][2]) * wordsize
            )
            for p in points
        ]
        if kernel_name[1:] == "gemm":
            y = [(2 * p[0][0] * p[0][1] * p[0][2] / p[1]) for p in points]
        elif kernel_name[1:] == "syrk":
            y = [(p[0][0] * p[0][1] * p[0][2] / p[1]) for p in points]
        plt.plot(x, y, label=name)

    peak_x = [x[0], x[-1]]
    peak_y = [128.0, 128.0]

    plt.plot(peak_x, peak_y, label="peak compute")
    plot_cache_boundries(128.0)

    plt.legend()

    plt.title(f"""{kernel_name}, params: {params}""")
    plt.ylabel("GFlops/sec")
    plt.xlabel("log2(bytes)")
    plt.xticks(range(int(x[0]), int(x[-1]), 2))
    fig_file_name = f"{kernel_graphs_dir}/peak_compute_{kernel_name}_{params}.png"
    fig_file_name = fig_file_name.replace(":", "=")  # For Windows compatibility
    plt.savefig(fig_file_name)


def ratio_and_gm_plot(params, names_to_points):
    names_list = list(names_to_points.keys())

    res = {"sum": 0, "iter": 0}
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            name1 = names_list[i]
            name2 = names_list[j]

            if "exo" not in name2 and "EXO" not in name2:
                name1, name2 = name2, name1
            if name2 != "Exo":
                continue

            points1 = names_to_points[name1]
            points2 = names_to_points[name2]
            sizes1 = [p[0] for p in points1]
            sizes2 = [p[0] for p in points2]
            runtimes1 = [p[1] for p in points1]
            runtimes2 = [p[1] for p in points2]

            # Points should have been ordered before
            if sizes1 != sizes2:
                print(
                    f"Could not plot ratios of {name1} against {name2}, benchmarks did not use the same sizes"
                )
                continue

            plt.clf()
            x = [log_2(sz) for sz in sizes1]
            y = [runtimes1[i] * 100 / runtimes2[i] for i in range(len(runtimes1))]
            plt.plot(x, y, label=f"{name1} runtime / {name2} runtime")

            plt.legend()

            GM = 1
            for r in y:
                GM *= r
            GM = GM ** (1 / len(sizes1))

            res["sum"] += GM
            res["iter"] += 1

            plt.title(
                f"""
                      {kernel_name}, params: {params}
                      GM = {GM}"""
            )
            plt.ylabel("runtime ratio % (ns)")
            plt.xlabel("log2(words)")
            plt.xticks(range(0, 30, 2))
            plt.yticks(list(range(0, 201, 20)) + list(range(250, 500, 50)))
            plt.axhline(y=100, color="r", linestyle="-")
            fig_file_name = (
                f"{kernel_graphs_dir}/ratios_{name1}_vs_{name2}_{params}.png"
            )
            fig_file_name = fig_file_name.replace(":", "=")  # For Windows compatibility
            plt.savefig(fig_file_name, bbox_inches="tight")

    # with open("summary.txt", "a") as f:
    #     line = f"{kernel_name}| {params}| {res['sum'] / res['iter']}\n"
    #     f.write(line)


for params in perf_res:
    if kernel_name[1:] in level3_kernels:
        peak_compute_plot(params, perf_res[params])
    else:
        ratio_and_gm_plot(params, perf_res[params])
        peak_bandwidth_plot(params, perf_res[params])

    # raw_runtime_plot(params, perf_res[params])
