#!/bin/bash
# Push a Kaggle job, wait for it, download the output, then ring until you press Ctrl-C.
# Usage: bash run_and_alarm.sh <kernel folder> <kaggle id, e.g. anasrazy/namma-final> <output dir, or - to skip the download> [file regex]
# Keeps the Mac awake while waiting (caffeinate). Keep the lid OPEN and the charger in: a closed lid still sleeps the Mac.
set -u
FOLDER="$1"; ID="$2"; OUT="$3"; PAT="${4:-}"   # PAT: download only matching files (Mac disk is small)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
caffeinate -dims -w $$ &
kaggle kernels push -p "$FOLDER" 2>&1 | grep -v Warning
START=$(date +%s)
while true; do
  s=$(kaggle kernels status "$ID" 2>&1 | grep -v Warning)
  echo "$(date '+%H:%M') $s"
  echo "$s" | grep -qiE "complete|error|cancel" && break
  sleep 60
done
if [ "$OUT" = "-" ]; then          # e.g. namma-prep: its output is a cache for other Kaggle jobs, nothing to download
  kaggle kernels output "$ID" -p /tmp/kaggle_log_only --file-pattern '.*\.log$' >/dev/null 2>&1 || true
else
  [ -d "$OUT" ] && mv "$OUT" "${OUT}_prev_$(date +%H%M)"
  kaggle kernels output "$ID" -p "$OUT" ${PAT:+--file-pattern "$PAT"} 2>&1 | grep -v Warning | tail -2
fi
MIN=$(( ($(date +%s) - START) / 60 ))
echo "$ID finished: $s  (${MIN} min). Output in $OUT"
[ "$OUT" = "-" ] || open "$OUT" 2>/dev/null
osascript -e "set volume output volume 100" -e "display notification \"$ID done ($s)\" with title \"NAMMA job finished\"" 2>/dev/null
echo "Ringing. Press Ctrl-C to stop."
while true; do afplay /System/Library/Sounds/Glass.aiff; say "Kaggle job finished"; sleep 1; done
