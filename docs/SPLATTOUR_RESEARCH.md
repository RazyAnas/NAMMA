# Splat Tour (splattour.com): how it works, from public sources (2026-10-22)

Everything below comes from public pages, the public tour viewer's own network requests, the public CDN files that the
viewer loads for every visitor, App Store / WHOIS / GitHub public APIs. No login, no private endpoints.

## Company
- Product: Splat Tour, "phone video -> 3D tour + floor plan + AI staging", Barcelona. Tour in 1-2 h. EUR 19-89/month.
- iPhone app "SplatTour: 3D Virtual Tour", released 2026-07-09; App Store developer/seller: **Jordi Pau Escursell Valero**.
- Domain registered 2026-04-25 (GoDaddy, privacy proxy). Email on Cloudflare.
- Processors named in their terms: **Modal Labs** (GPU compute), **Cloudflare R2** (storage), Stripe, Google OAuth.
- No GitHub account found under the founder's name.

## What a published tour actually is (public CDN files of the demo "Terrassa_Apartment")
- `configs/<tour>.json`: "load base_splat as fallback, swap to per-room splat when inside bbox".
  - one **overview splat** of the whole flat (`Tour_general.sog`, 4.2 MB) plus **one splat per room** (Bedroom, Hallway,
    Living_room, Office, WC: 4.6-11.5 MB each, SOG format). The viewer streams the room you are in.
  - per room: bbox, centroid, entry pose, 64 camera positions, a panorama rendered from the splat, a thumbnail.
  - a vector floor plan (walls, doors with swing/hinge, room labels) and 6 linked 360 panorama stations.
  - `<room>.init.json` points at `Tour_general/sparse/0`: **all rooms share one COLMAP model** (one coordinate frame).
- Viewer: Next.js site, **PlayCanvas engine** gsplat renderer, **SOG** compressed splats (PlayCanvas splat-transform).

## Their capture rules (public recording guide)
- Ultra-wide lens (0.5x), phone upright at chest height, aim at eye level, slow walk, stop before turning
  (">35 deg between frames breaks matching"), never orbit a point.
- Per room: lap along the walls, lap the other way, then both diagonals. Cross doorways through the middle.
- One uncut video per floor, 1080p/4K at 30 fps, SDR (no HDR), ~1 min per bathroom, ~2 min per big room.

## A public open repo with the same architecture: github.com/denlolsauce/splattour ("portal-pipeline")
Not proven to be Splat Tour's own code (different author handle; no license, so read for ideas, do not copy code).
It implements exactly the structure above, and its README lists the failures they hit:
- **Input: one overview video + one video per room**, each room video starting under the door frame.
  Exposure/focus locked; HDR, scene optimizer and "super steady" stabilization off.
- **Extract**: 2-8 fps; drop a frame only if it is below 0.6x the sharpness median of its +-15 neighbours
  (a global sharpness threshold deletes sharp frames of blank walls). Never drop more than 40%.
- **SfM**: one joint COLMAP 3.12.6 + **GLOMAP** run over all videos (shared frame); SIFT 8192, sequential matching with
  overlap 10 + loop detection + vocab tree matching. Rooms trained separately do not line up.
- **Train**: gsplat simple_trainer, **MCMC**, antialiased, **bilateral grid on**, no world-space normalisation;
  overview cap 400k Gaussians / 12k steps / SH1; each room 600k / 15k steps / SH2.
- **Compress**: PLY -> SOG; overview cut into per-room chunks by room bbox. **Nav**: doorway crossings = where two
  rooms' camera paths come within a threshold distance; Catmull-Rom smoothed walking path; floor height from cameras.
- Their #1 failure: **plain walls** -> tape up a poster during capture. Specular things (glass, rails) need extra angles.

## What this means for NAMMA
1. Our hardest problems (drift across a big space, blank walls, floaters off the path) are side-stepped, not solved:
   **small per-room splats in one shared SfM frame**, streamed by room. Each room is a short, dense, sharp capture.
2. Their capture discipline is the same lever our 30 s clip test proved (sharp, dense, slow): 21 dB -> 26.9 dB.
3. Cheap things to copy as ideas: relative (windowed) blur rejection, GLOMAP global mapper, MCMC + bilateral grid,
   per-room bbox streaming in the viewer, doorway detection from camera-path proximity.
