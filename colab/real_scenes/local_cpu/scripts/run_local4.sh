#!/bin/bash
# Local CPU off-path test, round 4: the mesh answer and a mesh-initialised splat, same held-out stretch.
export OS_BIN=$SCRATCH/os/opensplat-cpu; R=$SCRATCH/wt/colab/real_scenes; W=$SCRATCH/block/phone_sfm; W2=$SCRATCH/block/phone_mesh
python3 $R/scripts/mesh_offpath.py $W $W/mesh_boot 0.01 > $W/mesh_boot.log 2>&1; echo "mesh_boot $(tail -1 $W/mesh_boot.log | cut -c1-260)"
TEXTURE=4 python3 $R/scripts/mesh_offpath.py $W $W/mesh_tex 0.01 > $W/mesh_tex.log 2>&1; echo "mesh_tex $(tail -1 $W/mesh_tex.log | cut -c1-260)"
python3 $R/scripts/mesh_points.py $W/mesh_boot/mesh.ply $W/init_mesh_boot.ply 200000
mkdir -p $W2/clean && ln -sfn $W/data $W2/data && cp -r $W/clean/. $W2/clean/
python3 $R/scripts/make_whole_proj.py $W2 $W/init_mesh_boot.ply $W2/proj > /dev/null
L="--refine-every 100 --densify-from 500 --densify-until 2500"
RUN=b3k_meshinit TOTAL=3000 EXTRA="$L" bash $R/scripts/run_scene.sh $W2 > $W2/b3k_meshinit.log 2>&1; echo "b3k_meshinit $(cut -c1-220 $W2/b3k_meshinit/score.txt)"
echo LOCAL_BLOCK4_DONE
