#!/bin/bash
# Image-only camera solve of the 2026-10-20 lobby recording (all 16 AR pieces), COLMAP global mapper on CPU.
set -e; cd $SCRATCH/lobby_sfm; rm -f db.db; mkdir -p sparse
date; colmap feature_extractor --database_path db.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params 710,710,231,512 --FeatureExtraction.use_gpu 0 --SiftExtraction.max_num_features 4096 > fe.log 2>&1; echo FEATURES_DONE; date
colmap sequential_matcher --database_path db.db --SequentialMatching.overlap 15 --SequentialMatching.quadratic_overlap 1 --FeatureMatching.use_gpu 0 > sm.log 2>&1; echo SEQ_DONE; date
colmap matches_importer --database_path db.db --match_list_path loop_pairs.txt --match_type pairs --FeatureMatching.use_gpu 0 > lp.log 2>&1; echo LOOP_DONE; date
colmap global_mapper --database_path db.db --image_path images --output_path sparse > gm.log 2>&1; echo MAPPER_DONE; date
ls sparse; for d in sparse/*; do colmap model_analyzer --path $d 2>&1 | grep -E "Cameras|Images|Registered|Points|reprojection" ; done
