import json
import sys
import matplotlib.pyplot as plt
import math
import os

vectors = {"snrm2" : 1, "sscal": 1, "scopy": 2, "srot":2, "sswap": 2, "sasum": 1, "sdot":2}

if len(sys.argv) != 5:
    print("python grpah.py <kernel name> <google benchmark output json file> <BLA_VENDOR> <graphs_dir>!")
    exit(1)

kernel_name = sys.argv[1]
BLA_VENDOR = sys.argv[3]
graphs_dir = sys.argv[4]

if not os.path.exists(graphs_dir):
    os.mkdir(graphs_dir)

kernel_graphs_dir = graphs_dir + "/" + kernel_name
if not os.path.exists(kernel_graphs_dir):
    os.mkdir(kernel_graphs_dir)

with open(sys.argv[2]) as f:
    data = json.load(f)

# TODO: We should probably use this context data for time tracking
context = data['context']

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
for d in data['benchmarks']:
    lis = d['name'].split('/')
    name = lis[0].split('_')
    name = " ".join(name[1:])

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
        return (size*vectors.get(kernel_name, 2)*10**9)/(time*2**30)

    plt.clf()
    for name in names_to_points:
        points = names_to_points[name]
        x = [log_2(p[0]) for p in points]
        y = [get_gword_sec(p[0], p[1]) for p in points]
        plt.plot(x, y, label=name)
    
    peak_x = [0, log_2(32*1024/4), log_2(256*1024/4), log_2(6*1024*1024/4), log_2(66*1024*1024/4)]
    peak_y = [60.764375, 26.390875, 14.170275, 8.750275, 8.750275]
    plt.plot(peak_x, peak_y, drawstyle='steps-post', label="peak bandwidth")
    
    plt.legend()
    
    plt.title(kernel_name + ", params: " + str(params) + "\nBLAS Reference: " + BLA_VENDOR + ", kernel with AVX2")
    plt.ylabel('Gwords/sec')
    plt.xlabel('log(words)')
    plt.savefig(kernel_graphs_dir + "/" + "peak_bandwidth_" + kernel_name + str(params) + '.png')
    
for params in perf_res:
    peak_bandwidth_plot(params, perf_res[params])
