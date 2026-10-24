# nav_from_mesh: collision mesh -> floor, walkable grid, floor plan, A* path (CPU, ~0.5 s/scene)

`python nav_from_mesh.py <collision.glb> <out_prefix>`; floor height and up axis are estimated from the mesh alone
(strongest horizontal layer in the lower 40% of the height range). Walkable = floor seen, no obstacle 0.15-1.7 m above
floor, 0.25 m clearance, largest connected region. Run on the three `walk_demo` collision meshes, 5 cm cells (measured):

| scene | floor seen m2 | walkable m2 | A* corner-to-corner | time |
|---|---|---|---|---|
| lobby | 20.9 | 15.2 | 8.75 m | 0.44 s + 0.03 s |
| icl_room | 25.7 | 14.4 | 5.76 m | 0.40 s + 0.004 s |
| phone_room | 8.0 | **0.99** | 2.15 m | 0.51 s |

Looked at the plots (`*.png`): lobby and icl_room look right (two obstacles in the lobby, room outline + furniture in icl_room).
**phone_room fails**: nearly everything is marked obstacle; the floor layer or the 1.7 m head band is probably wrong for that
tilted/partial mesh. Not yet debugged. Lobby mesh is only 1.2 m tall, so the head band means little there.
Not tested against ground truth; "plausible by eye" only.
