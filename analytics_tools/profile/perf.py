import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_PATH = Path(__file__)
PROFILE_DIR = SCRIPT_PATH.parent.resolve()

# Events to collect
events = [
    "branch-instructions",
    "branch-misses",
    "cache-misses",
    "cache-references",
    "instructions",
    "cpu-cycles",
    "page-faults",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L1-dcache-stores",
    "L1-icache-load-misses",
    "l2_rqsts.all_demand_data_rd",
    "l2_rqsts.demand_data_rd_hit",
    "l2_rqsts.demand_data_rd_miss",
    "LLC-loads",
    "LLC-load-misses",
    "LLC-stores",
    "LLC-store-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "dTLB-stores",
    "dTLB-store-misses",
    "iTLB-load-misses",
]

# Function to run the binary with perf stat
def run_perf(binary_path, fltr, size):
    cmd = ["perf", "stat"]
    for event in events:
        cmd.extend(["-e", event])
    cmd.extend([binary_path, f"--benchmark_filter={fltr}{size}"])

    # Execute the command
    print("Running ", cmd)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stderr


def parse_perf_output(output):
    data = {}
    print("Parsing")
    print(output)
    # Regex to match the perf stat output lines
    pattern = re.compile(r"(\d+[,\d+]*)\s+.*\s+([\.\w-]+)(?::u).*")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            event_name = match.group(2)
            count = int(match.group(1).replace(",", ""))
            if event_name in events:
                data[event_name] = count
    print(data)
    return data


def plot(sizes, results):
    # Plotting results
    fig, axes = plt.subplots(len(events), 1, figsize=(10, 50))
    for idx, event in enumerate(events):
        for fltr, result in results:
            axes[idx].plot(sizes, result[event], label=fltr, marker="o")
        axes[idx].set_title(event)
        axes[idx].set_xlabel("Input Size")
        axes[idx].set_ylabel("Count")
        axes[idx].grid(True)
        axes[idx].set_xscale("log")
        axes[idx].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(PROFILE_DIR / "perf.png")


# Main function to collect and plot data
def main(binary_path, fltrs, sizes):
    results = []

    for fltr in fltrs:
        result = {event: [] for event in events}

        # Collect data
        for size in sizes:
            output = run_perf(binary_path, fltr, size)
            data = parse_perf_output(output)
            for event in events:
                value = data.get(event, 0)
                result[event].append(value)

        results.append(result)

    plot(sizes, list(zip(fltrs, results)))


# Example usage
if __name__ == "__main__":
    sizes = [i for i in range(500, 300, 100)]
    fltrs = ["exo_sgemm_rm_nn/M:"]
    binary = "/home/samir/ExoBLAS/build/avx512/test/level3/gemm_bench"
    main(binary, fltrs, sizes)
