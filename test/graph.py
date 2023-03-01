import json
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("python grpah.py <kernel name> <google benchmark output json file>!")

kernel_name = sys.argv[1]

with open(sys.argv[2]) as f:
    data = json.load(f)

# TODO: We should probably use this context data for time tracking
context = data['context']

perf_res = {}
for d in data['benchmarks']:
    lis = d['name'].split('/')
    name = lis[0]
    size = lis[1]
    time = d['cpu_time']
    p = perf_res.setdefault( name, ([size], [time]) )
    p[0].append(size)
    p[1].append(time)

for key in perf_res:
    plt.plot(perf_res[key][0], perf_res[key][1], label=key)
    plt.legend()

plt.ylabel('runtime (ns)')
plt.xlabel('size')
plt.savefig(kernel_name+'.png')
