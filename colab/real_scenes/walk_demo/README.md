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
Fly mode stays within 0.6 m of the capture path when the room has `cameras.json` (camera centres from `scenes/<scene>/clean/poses_w2c.json`): a splat is only reliable near where the phone filmed.
Hybrid view: when a room has `mesh.glb` (`../scripts/make_demo_mesh.py`: TSDF of the anchored depth, 300k triangles,
vertex colours, then `npx gltfpack -cc -vp 14`: 6.1 MB to 1.2 MB), the mesh is drawn behind the splat. M cycles fill (default: mesh only plugs the splat's holes) / auto (fade
to the mesh 0.15-0.6 m from the capture path) / splat / mesh. Off the path the mesh scores 18.8 dB vs 15.3-16.6 for the splats.
Floors are built by `../scripts/build_walk_scene.py`.
