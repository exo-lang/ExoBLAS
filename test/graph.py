import json
import sys
import matplotlib.pyplot as plt
import math
import os


def mem_footprint(kernel_name, size):
    if kernel_name in ["snrm2", "sscal", "scopy", "srot", "sswap", "sasum", "sdot", "saxpy"]:
        return size
    elif kernel_name in ["sgemv"]:
        return size*size
    else:
        raise NotImplementedError(f"Input size of {kernel_name} is not implemented")


def mem_ops(kernel_name, size):
    """
    Returns total memory usage of `kernel_name` of dimension `size` in words.
    """
    mem_ops = {
        1: {"snrm2" : 1, "sscal": 1, "scopy": 2, "srot":2, "sswap": 2, "sasum": 1, "sdot":2, "saxpy": 2},
        2: {"sgemv": 1}
    }

    if kernel_name in mem_ops[1].keys():
        return size*mem_ops[1].get(kernel_name, 2)
    elif kernel_name in mem_ops[2].keys():
        # TODO: generalize beyond nxn matrices
        return size*size*mem_ops[2].get(kernel_name, 1)
    else:
        raise NotImplementedError(f"Memory usage of {kernel_name} is not implemented")


if len(sys.argv) != 4:
    print("python graph.py <kernel name> <benchmark results dir> <graphs_dir>!")
    exit(1)

kernel_name = sys.argv[1]
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
    json_path = f"{benchmark_results_dir}/{dir}/{kernel_name}.json"
    if not os.path.exists(json_path):
        continue
    
    with open(json_path) as f:
        data = json.load(f)
    jsons_dict[name] = data

def log_2(x):
    return math.log(x, 2)

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
    for d in data['benchmarks']:
        lis = d['name'].split('/')
        if "Fixture" in lis[0]:
            lis = lis[1:]
        name = benchmark_name
        size = int(lis[1].split(":")[1]) # Words
        params = tuple(lis[2:])
        time = d['cpu_time'] # ns
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
    def get_gword_sec(size, time):
        return (mem_ops(kernel_name, size)*10**9)/(time*2**30)

    plt.clf()
    for name in names_to_points:
        points = names_to_points[name]
        x = [log_2(mem_footprint(kernel_name, p[0])) for p in points]
        y = [get_gword_sec(p[0], p[1]) for p in points]
        plt.plot(x, y, label=name)
    
    peak_x = [0, log_2(32*1024/4), log_2(256*1024/4), log_2(6*1024*1024/4), log_2(66*1024*1024/4)]
    peak_y = [60.764375, 26.390875, 14.170275, 8.750275, 8.750275]

    if x[-1] > peak_x[-1]:
        peak_x[-1] = x[-1]

    plt.plot(peak_x, peak_y, drawstyle='steps-post', label="peak bandwidth")
    
    plt.legend()
    
    plt.title(f"""{kernel_name}, params: {params}""")
    plt.ylabel('Gwords/sec')
    plt.xlabel('log(words)')
    plt.xticks(range(0, 30, 2))
    fig_file_name = f"{kernel_graphs_dir}/peak_bandwidth_{kernel_name}_{params}.png"
    fig_file_name = fig_file_name.replace(":", "=") # For Windows compatibility
    plt.savefig(fig_file_name)

def raw_runtime_plot(params, names_to_points):
    plt.clf()
    for name in names_to_points:
        points = names_to_points[name]
        x = [log_2(p[0]) for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, label=name)
    
    plt.legend()
    
    plt.title(f"""{kernel_name}, params: {params}""")
    plt.ylabel('runtime (ns)')
    plt.xlabel('log(words)')
    plt.xticks(range(0, 30, 2))
    fig_file_name = f"{kernel_graphs_dir}/raw_runtime_{kernel_name}_{params}.png"
    fig_file_name = fig_file_name.replace(":", "=") # For Windows compatibility
    plt.savefig(fig_file_name, bbox_inches='tight')

def ratio_and_gm_plot(params, names_to_points):
    names_list = list(names_to_points.keys())
    
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            name1 = names_list[i]
            name2 = names_list[j]
            
            if "exo" not in name2 and "EXO" not in name2:
                name1, name2 = name2, name1
            
            points1 = names_to_points[name1]
            points2 = names_to_points[name2]
            sizes1 = [p[0] for p in points1]
            sizes2 = [p[0] for p in points2]
            runtimes1 = [p[1] for p in points1]
            runtimes2 = [p[1] for p in points2]
            
            # Points should have been ordered before
            if sizes1 != sizes2:
                print(f"Could not plot ratios of {name1} against {name2}, benchmarks did not use the same sizes")
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
            
            plt.title(f"""
                      {kernel_name}, params: {params}
                      GM = {GM}""")
            plt.ylabel('runtime ratio % (ns)')
            plt.xlabel('log(words)')
            plt.xticks(range(0, 30, 2))
            plt.yticks(list(range(0, 201, 20)) + list(range(250, 500, 50)))
            plt.axhline(y = 100, color = 'r', linestyle = '-')
            fig_file_name = f"{kernel_graphs_dir}/ratios_{name1}_vs_{name2}_{params}.png"
            fig_file_name = fig_file_name.replace(":", "=") # For Windows compatibility
            plt.savefig(fig_file_name, bbox_inches='tight')
    
for params in perf_res:
    peak_bandwidth_plot(params, perf_res[params])
    raw_runtime_plot(params, perf_res[params])
    ratio_and_gm_plot(params, perf_res[params])