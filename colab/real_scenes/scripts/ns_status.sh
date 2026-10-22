#!/bin/bash
# One-screen status of a kaggle_nerfstudio_baseline.sh run: usage ns_status.sh <NAME> (work dir /content/<NAME>)
W=/content/${1:-ns_clip30}
date +%H:%M:%S; tail -6 /kaggle/working/${1:-clip30}.log 2>/dev/null | cut -c1-110
[ -f $W/process.log ] && tr '\r' '\n' < $W/process.log | grep -vE '^\s*$' | tail -1 | cut -c1-110
for f in $W/train_*.log; do [ -f $f ] && echo "$(basename $f): $(tr '\r' '\n' < $f | grep -E '^ *[0-9]+ \(' | tail -1 | cut -c1-80)"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
