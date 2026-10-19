# NAMMA walking demo, three rooms

Static files, no build step. Serve this folder (`python3 -m http.server 8000`) and open http://localhost:8000.
Pick a room on the start screen (or `?scene=phone|lobby|icl`); the "rooms" link in the corner goes back.

| Room | Splat | Walking floor and collision |
|---|---|---|
| `data/phone_room` | phone video, our SfM poses, 30k steps, 35.53 dB held out | our anchored monocular depth; floor = RANSAC plane 1.5 m below the cameras |
| `data/lobby` | lobby video, merged COLMAP poses, 30k steps, 23.16 dB held out | flat floor 1.45 m below the camera path, widened 0.6 m (its depth has no usable floor) |
| `data/icl_room` | ICL-NUIM living room, 3k steps CPU, 32.26 dB held out | the dataset's depth |

The phone-room and lobby `splat.ply` files are not in git (90 and 70 MB, trained on Colab): they come from
`namma_phone_sfm_lobby_30k.zip` (`phone_sfm_run30k_splat.ply`, `lobby_ref_run30k_splat.ply`).
Floors are built by `../scripts/build_walk_scene.py`.
