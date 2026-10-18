#!/bin/bash
# usage: run_splat.sh <proj dir> <out dir>  (today's whole-room recipe: 3000 steps, 1/4 res to 500, 1/2 to 1000, full after)
P=$1; out=$2; mkdir -p $out; date +%s > $out/start.txt
OPENSPLAT_TEST_PREFIX=test_ /tmp/claude-0/work/OpenSplat/build2/opensplat $P --cpu -n 3000 -d 1 --num-downscales 2 --resolution-schedule 500 --val-image test_00032.jpg --val-render $out -o $out/splat.ply > $out/log.txt 2>&1
date +%s > $out/end.txt
python3 /mnt/project-files/wholeroom/scripts/score_whole.py $out /tmp/claude-0/work/data/namma_wholeroom_gpu/data/images/clean $out/scores.json > $out/score.txt 2>&1
