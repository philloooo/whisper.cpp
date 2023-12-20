#!/bin/bash

powermetrics -i 1000 -s cpu_power  >scripts/power_result.log &
powerPID=$!

./main -f scripts/librispeech-concat.wav -m models/ggml-tiny.bin 2>&1 | grep "total time"
kill $powerPID
