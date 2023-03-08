import json
import sys
import matplotlib.pyplot as plt
import math

vectors = {"snrm2" : 1, "sscal": 1, "scopy": 2, "srot":2, "sswap": 2, "sasum": 1, "sdot":2}

if len(sys.argv) != 3:
    print("python grpah.py <kernel name> <google benchmark output json file>!")

kernel_name = sys.argv[1]
plt.title(kernel_name + " kernel with AVX2")

with open(sys.argv[2]) as f:
    data = json.load(f)

# TODO: We should probably use this context data for time tracking
context = data['context']

perf_res = {}
for d in data['benchmarks']:
    lis = d['name'].split('/')
    name = lis[0].split('_')
    name = " ".join(name[1:])
    size = int(lis[1]) # Words
    time = d['cpu_time'] # ns
    # size : word already
    # time is nanoseconds
    gword_sec = (size*vectors.get(kernel_name, 2)*10**9)/(time*2**30) # (Words*1 / ns) => Gwords/s
    size = math.log(size)
    #print(size, time, gword_sec)
    p = perf_res.setdefault( name, ([size], [gword_sec]) )
    p[0].append(size)
    p[1].append(gword_sec)

for key in perf_res:
    sorted_points = sorted([(perf_res[key][0][i], perf_res[key][1][i]) for i in range(len(perf_res[key][0]))])
    x = [p[0] for p in sorted_points]
    y = [p[1] for p in sorted_points]
    plt.plot(x, y, label=key)
    plt.legend()

f = math.log
peak_x = [0, f(32*1024/4), f(256*1024/4), f(6*1024*1024/4), f(66*1024*1024/4)]
peak_y = [60.764375, 26.390875, 14.170275, 8.750275, 8.750275]
plt.plot(peak_x, peak_y, drawstyle='steps-post', label="peak bandwidth")
plt.legend()

plt.ylabel('Gwords/sec')
plt.xlabel('log(words)')
plt.savefig(kernel_name+'.png')
