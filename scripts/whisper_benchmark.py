
import subprocess
import re
import numpy
import tempfile
import time
import datetime


def run_whisper_command(audio_input):
    cmd = ["./main", "-f", audio_input, "-m", "models/ggml-tiny.bin"]
    result = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result


def run_whisper(data):
    result = run_whisper_command("samples/jfk.wav")
    assert (result.stdout ==
            b'\n[00:00:00.000 --> 00:00:10.560]   And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.\n\n')
    matching = "whisper_print_timings: +{} time = +(\d+\.\d+)"
    for line in result.stderr.decode('utf-8').split('\n'):
        encode = re.search(matching.format("encode"), line)
        if encode:
            data["encode"].append(float(encode.group(1)))
            continue
        decode = re.search(matching.format("decode"), line)
        if decode:
            data["decode"].append(float(decode.group(1)))
            continue
        load = re.search("Core ML model load time: +(\d+\.\d+)", line)
        if load:
            data["load coreml"].append(float(load.group(1)))
            continue
        total = re.search(matching.format("total"), line)
        if total:
            data["total"].append(float(total.group(1)))
    return data


def collect_metrics(num_of_runs=10):
    data = {'encode': [], 'decode': [], 'load coreml': [], 'total': []}
    for i in range(num_of_runs):
        run_whisper(data)
    print(data)
    for k, v in data.items():
        if v:
            print("{} time: first run: {:0.2f}, avg: {:0.2f}, std: {:0.2f}".format(
                k, v[0], numpy.average(v[1:]), numpy.std(v[1:])))
    return data


def parse_power_metrics(log):
    result = []
    time_delta = 0
    with open(log, "r") as fd:
        for line in fd.readlines():
            # 1s sampling rate, but actual sampling rate varies, but it records the elapsed time.
            elapsed = re.search("(\d+\.\d+)ms elapsed", line)
            if elapsed:
                result.append({"time_delta": float(elapsed.group(1))})
                continue
            gpu = re.search("GPU Power: (\d+) mW", line)
            if gpu:
                result[-1]["GPU"] = int(gpu.group(1))
                continue
            cpu = re.search("CPU Power: (\d+) mW", line)
            if cpu:
                result[-1]["CPU"] = int(cpu.group(1))
                continue
            ane = re.search("ANE Power: (\d+) mW", line)
            if ane:
                result[-1]["ANE"] = int(ane.group(1))
                continue
    area = {"CPU": 0, "GPU": 0, "ANE": 0}
    prev = {"CPU": 0, "GPU": 0, "ANE": 0}
    total_time = 0
    for item in result:
        if not ("CPU" in item and "GPU" in item and "ANE" in item):
            break
        total_time += item["time_delta"]
        for target in area:
            # Trapezoid area
            area[target] += 0.5 * item["time_delta"] * \
                (item[target] + prev[target])
        prev = item
    # mW, ms -> W ,s
    for key in area:
        area[key] = round(area[key]/1000/1000, 2)
    return {"raw": result, "area": area, "total_time": round(total_time/1000, 2)}


def main():
    result = parse_power_metrics("scripts/power_result.log")
    print(result["area"])
    print("total time: {}".format(result["total_time"]))
    return result


main()
