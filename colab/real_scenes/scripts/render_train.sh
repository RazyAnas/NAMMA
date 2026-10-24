#!/bin/bash
# Render every training view of a splat, in chunks (as render_train.sh in the CPU pipeline).
# usage: render_train.sh <work> <splat.ply> <out dir> [chunk=30]
D=$(dirname $0); W=$1; PLY=$2; OUT=$3; CH=${4:-30}; mkdir -p $OUT
STEP=$(head -c 200 $PLY | grep -a -o 'iteration [0-9]*' | awk '{print $2}')
while true; do
  MISS=$(python3 -c "
import json,os; st=json.load(open('$W/clean/stats.json'))
m=[f for f in st['train_frames'] if not os.path.exists('$OUT/test_%05d.png'%f)][:$CH]; print(','.join(map(str,m)))")
  [ -z "$MISS" ] && break
  python3 $D/make_render_proj.py $W $W/proj_rchunk $MISS > /dev/null
  OPENSPLAT_TEST_PREFIX=test_ $OS_BIN $W/proj_rchunk -n $STEP -d 1 --num-downscales 2 \
    --resolution-schedule 500 --resume $PLY --val-render $OUT -o $OUT/x.ply >> $OUT/log.txt 2>&1 || { echo fail; exit 1; }
done
rm -f $OUT/x.ply; echo done
