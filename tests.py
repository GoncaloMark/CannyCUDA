import subprocess
import re
import statistics

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e.output)
        return ""

def extract_times(output):
    host_time = device_time = None
    host_match = re.search(r"Host processing time:\s*([\d.]+)", output)
    device_match = re.search(r"Device processing time:\s*([\d.]+)", output)

    if host_match:
        host_time = float(host_match.group(1))
    if device_match:
        device_time = float(device_match.group(1))

    return host_time, device_time

def run_multiple(cmd, count):
    host_times, device_times = [], []
    for i in range(count):
        print(f"Running: {cmd} [{i+1}/{count}]")
        output = run_command(cmd)
        host, device = extract_times(output)
        if host is not None and device is not None:
            host_times.append(host)
            device_times.append(device)
    return host_times, device_times

def summarize(times):
    return {
        "average": statistics.mean(times),
        "stddev": statistics.stdev(times),
        "median": statistics.median(times)
    }

canny_host, canny_device = run_multiple("./canny", 5)
canny_t_host, canny_t_device = run_multiple("./canny -t in.txt", 5)

canny_host_stats = summarize(canny_host)
canny_device_stats = summarize(canny_device)
canny_t_host_stats = summarize(canny_t_host)
canny_t_device_stats = summarize(canny_t_device)

with open("results.txt", "w") as f:
    f.write("== ./canny (5 runs) ==\n")
    f.write(f"Host times: {canny_host}\n")
    f.write(f"Host Avg: {canny_host_stats['average']:.3f} ms\n")
    f.write(f"Host StdDev: {canny_host_stats['stddev']:.3f} ms\n")
    f.write(f"Host Median: {canny_host_stats['median']:.3f} ms\n")
    f.write(f"Device times: {canny_device}\n")
    f.write(f"Device Avg: {canny_device_stats['average']:.3f} ms\n")
    f.write(f"Device StdDev: {canny_device_stats['stddev']:.3f} ms\n")
    f.write(f"Device Median: {canny_device_stats['median']:.3f} ms\n")
    speedup = canny_host_stats['average'] / canny_device_stats['average']
    f.write(f"Speedup (Host/Device): {speedup:.2f}x\n\n")

    f.write("== ./canny -t in.txt (5 runs) ==\n")
    f.write(f"Host times: {canny_t_host}\n")
    f.write(f"Host Avg: {canny_t_host_stats['average']:.3f} ms\n")
    f.write(f"Host StdDev: {canny_t_host_stats['stddev']:.3f} ms\n")
    f.write(f"Host Median: {canny_t_host_stats['median']:.3f} ms\n")
    f.write(f"Device times: {canny_t_device}\n")
    f.write(f"Device Avg: {canny_t_device_stats['average']:.3f} ms\n")
    f.write(f"Device StdDev: {canny_t_device_stats['stddev']:.3f} ms\n")
    f.write(f"Device Median: {canny_t_device_stats['median']:.3f} ms\n")
    speedup = canny_t_host_stats['average'] / canny_t_device_stats['average']
    f.write(f"Speedup (Host/Device): {speedup:.2f}x\n")

print("All results written to results.txt.")