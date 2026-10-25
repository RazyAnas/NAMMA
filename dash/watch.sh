#!/bin/bash
# Live dashboard on http://localhost:8765 . Usage: bash watch.sh   (Ctrl-C to stop). Needs the kaggle CLI already set up.
# Plan/progress comes from GitHub (updated by Claude with each push); job status comes from your own kaggle CLI every 60 s.
D=~/NAMMA/dash; mkdir -p $D; cd $D
B=https://raw.githubusercontent.com/RazyAnas/NAMMA/claude/wonderful-cori-i3fn41/dash
curl -fsSo index.html $B/index.html
JOBS="anasrazy/namma-ibr anasrazy/namma-train anasrazy/namma-prep anasrazy/namma-resfm"
python3 -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 & SRV=$!; trap "kill $SRV" EXIT
open http://localhost:8765 2>/dev/null; echo "dashboard: http://localhost:8765"
while true; do
  rows=""; for j in $JOBS; do s=$(kaggle kernels status $j 2>&1 | grep -oE 'RUNNING|COMPLETE|ERROR|QUEUED|CANCEL[A-Z_]*' | head -1); rows="$rows{\"id\":\"$j\",\"status\":\"${s:-?}\"},"; done
  L=$(ls -t ~/NAMMA/*_out*/*.log 2>/dev/null | head -1); T=""; [ -n "$L" ] && T=$(echo "== $L"; tail -n 15 "$L")
  python3 -c 'import json,sys;print(json.dumps({"at":sys.argv[1],"jobs":json.loads("["+sys.argv[2].rstrip(",")+"]"),"log":sys.argv[3]}))' "$(date +%H:%M:%S)" "$rows" "$T" > kaggle.json.tmp && mv kaggle.json.tmp kaggle.json
  sleep 60
done
