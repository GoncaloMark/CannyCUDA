import subprocess
import re
import statistics

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running command: {}".format(cmd))
        print(e.output)
        return ""

def extract_times(output, has_optimized=False):
    host_time = None
    device_time = None
    optimized_device_time = None

    host_match = re.search(r"Host processing time:\s*([\d.]+)", output)
    device_match = re.search(r"Device processing time:\s*([\d.]+)", output)
    if has_optimized:
        optimized_device_match = re.search(r"Device \(Speed Optimized\) processing time:\s*([\d.]+)", output)

    if host_match:
        host_time = float(host_match.group(1))
    if device_match:
        device_time = float(device_match.group(1))
    if has_optimized and optimized_device_match:
        optimized_device_time = float(optimized_device_match.group(1))

    if has_optimized:
        return host_time, device_time, optimized_device_time
    else:
        return host_time, device_time

def run_multiple(cmd, count, has_optimized=False):
    host_times, device_times, optimized_device_times = [], [], []
    for i in range(count):
        print(f"Running: {cmd} [{i+1}/{count}]")
        output = run_command(cmd)
        if has_optimized:
            host, device, optimized = extract_times(output, has_optimized=True)
            if host is not None and device is not None and optimized is not None:
                host_times.append(host)
                device_times.append(device)
                optimized_device_times.append(optimized)
        else:
            host, device = extract_times(output)
            if host is not None and device is not None:
                host_times.append(host)
                device_times.append(device)
    if has_optimized:
        return host_times, device_times, optimized_device_times
    else:
        return host_times, device_times

def summarize(times):
    return {
        "average": statistics.mean(times),
        "stddev": statistics.stdev(times),
        "median": statistics.median(times)
    }

canny_host, canny_device = run_multiple("./canny", 5)
canny_m_host, canny_m_device = run_multiple("./canny -m", 5)
canny_t_host, canny_t_device = run_multiple("./canny -t in.txt", 5)
canny_f_host, canny_f_device, canny_f_device_optimized = run_multiple("./canny -f", 5, has_optimized=True)

canny_host_stats = summarize(canny_host)
canny_device_stats = summarize(canny_device)
canny_m_host_stats = summarize(canny_m_host)
canny_m_device_stats = summarize(canny_m_device)
canny_t_host_stats = summarize(canny_t_host)
canny_t_device_stats = summarize(canny_t_device)
canny_f_host_stats = summarize(canny_f_host)
canny_f_device_stats = summarize(canny_f_device)
canny_f_device_optimized_stats = summarize(canny_f_device_optimized)

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
    
    f.write("== ./canny -m (5 runs) ==\n")
    f.write(f"Host times: {canny_m_host}\n")
    f.write(f"Host Avg: {canny_m_host_stats['average']:.3f} ms\n")
    f.write(f"Host StdDev: {canny_m_host_stats['stddev']:.3f} ms\n")
    f.write(f"Host Median: {canny_m_host_stats['median']:.3f} ms\n")
    f.write(f"[SM] Device times: {canny_m_device}\n")
    f.write(f"[SM] Device Avg: {canny_m_device_stats['average']:.3f} ms\n")
    f.write(f"[SM] Device StdDev: {canny_m_device_stats['stddev']:.3f} ms\n")
    f.write(f"[SM] Device Median: {canny_m_device_stats['median']:.3f} ms\n")
    speedup = canny_m_host_stats['average'] / canny_m_device_stats['average']
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
    f.write(f"Speedup (Host/Device): {speedup:.2f}x\n\n")

    f.write("== ./canny -f (5 runs) ==\n")
    f.write(f"Host times: {canny_f_host}\n")
    f.write(f"Host Avg: {canny_f_host_stats['average']:.3f} ms\n")
    f.write(f"Host StdDev: {canny_f_host_stats['stddev']:.3f} ms\n")
    f.write(f"Host Median: {canny_f_host_stats['median']:.3f} ms\n")
    f.write(f"Device times: {canny_f_device}\n")
    f.write(f"Device Avg: {canny_f_device_stats['average']:.3f} ms\n")
    f.write(f"Device StdDev: {canny_f_device_stats['stddev']:.3f} ms\n")
    f.write(f"Device Median: {canny_f_device_stats['median']:.3f} ms\n")
    f.write(f"Device (Speed Optimized) times: {canny_f_device_optimized}\n")
    f.write(f"Device (Speed Optimized) Avg: {canny_f_device_optimized_stats['average']:.3f} ms\n")
    f.write(f"Device (Speed Optimized) StdDev: {canny_f_device_optimized_stats['stddev']:.3f} ms\n")
    f.write(f"Device (Speed Optimized) Median: {canny_f_device_optimized_stats['median']:.3f} ms\n")
    speedup = canny_f_host_stats['average'] / canny_f_device_optimized_stats['average']
    f.write(f"Speedup (Host/Optimized Device): {speedup:.2f}x\n")

print("All results written to results.txt.")
