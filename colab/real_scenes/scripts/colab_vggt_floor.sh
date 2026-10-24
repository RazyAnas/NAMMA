#!/bin/bash
# Hybrid floors on a Colab GPU: VGGT-1B depth, anchored to our SfM points with the same code as the DA-V2 boot depth,
# feeds NAMMA's collision + NavMesh builder for the walking demo (the splat keeps DA-V2). Writes /content/vggt_floors.zip.
# usage: bash colab_vggt_floor.sh "phone_sfm lobby_ref"   (from the cloned NAMMA repo)
R=$(cd $(dirname $0)/..; pwd); SCENES=${1:-"phone_sfm lobby_ref"}; cd /content
pip -q install git+https://github.com/facebookresearch/vggt.git open3d trimesh pycolmap pydantic-settings imageio-ffmpeg plyfile > /tmp/pip_vggt.log 2>&1
[ -d /content/namma_app ] || git clone -q --depth 1 -b claude/project-thread-pd8o1z https://github.com/RazyAnas/NAMMA /content/namma_app
mkdir -p /content/vggt_floors
for s in $SCENES; do
  f=phone_frames; [ $s = lobby_ref ] && f=lobby_frames; W=/content/vwork/$s
  mkdir -p $W/data/color_raw $W/clean; cp $R/frames/$f/*.jpg $W/data/color_raw/; cp -r $R/scenes/$s/clean/. $W/clean/
  DEPTH_SRC=vggt python3 $R/scripts/prep_depth.py $W - 2>&1 | grep -v Warning | tail -2
  for d in vggt_boot_depth boot_depth; do
    o=/content/vggt_floors/${s}_$d; DEPTH_DIR=$d python3 $R/scripts/build_walk_scene.py $W /content/namma_app $o 2>&1 | grep -v "WARN\|Extract\|bad average" | tail -1
    [ -f $o/geometry/navigation/navmesh.json ] && python3 $R/scripts/nav_top.py $o/geometry $o/top.png
  done
done
cd /content/vggt_floors && zip -qr /content/vggt_floors.zip . -x '*/dense/*' '*/collision-preview.ply'; ls -la /content/vggt_floors.zip; echo VGGT_DONE
