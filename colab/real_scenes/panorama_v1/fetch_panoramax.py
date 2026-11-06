"""Fetch a street-level 360 route from Panoramax (open API, no key) for testing.

Panoramax is the open federated street-imagery commons (api.panoramax.xyz).
Car-mounted 360 sequences there look like Street View, so they let the
pipeline be tested end to end without a Google key. Files are written as
NNNN_<id>.jpg in route order plus manifest.json (lat, lon, heading, date,
licence), the same manifest shape the selector produces.
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

API = "https://api.panoramax.xyz/api"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "namma-panorama-test/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _dist_m(a, b) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    dx = (b[0] - a[0]) * 111_320 * math.cos(lat)
    dy = (b[1] - a[1]) * 110_540
    return math.hypot(dx, dy)


def collection_items(collection: str, max_items: int = 3000) -> list[dict]:
    items, url = [], f"{API}/collections/{collection}/items?limit=500"
    while url and len(items) < max_items:
        page = _get(url)
        items += page.get("features", [])
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
    items.sort(key=lambda f: f["properties"].get("datetime", ""))
    return items


def pick_route(items: list[dict], count: int = 40, spacing_m: float = 10.0, start: int = 0) -> list[dict]:
    picked, last = [], None
    for f in items[start:]:
        xy = f["geometry"]["coordinates"]
        if last is None or _dist_m(last, xy) >= spacing_m:
            picked.append(f)
            last = xy
            if len(picked) >= count:
                break
    return picked


def fetch(collection: str, out_dir: Path, count: int = 40, spacing_m: float = 10.0, start: int = 0,
          asset: str = "hd", log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = collection_items(collection)
    route = pick_route(items, count, spacing_m, start)
    if len(route) < 8:
        raise RuntimeError(f"Only {len(route)} panoramas at {spacing_m} m spacing in {collection}")
    manifest = {"schema_version": 1, "scene_name": f"Panoramax {collection}",
                "source": {"provider": "panoramax", "collection": collection, "api": API}, "panoramas": []}
    for i, f in enumerate(route):
        p = f["properties"]
        name = f"{i:04d}_{f['id'][:8]}.jpg"
        dest = out_dir / name
        if not dest.exists():
            req = urllib.request.Request(f["assets"][asset]["href"], headers={"User-Agent": "namma-panorama-test/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
        lon, lat = f["geometry"]["coordinates"][:2]
        manifest["panoramas"].append({
            "pano_id": f["id"], "image": name, "sequence": i, "lat": lat, "lon": lon,
            "heading_deg": p.get("view:azimuth"), "datetime": p.get("datetime"),
            "license": p.get("license"), "camera": (p.get("pers:interior_orientation") or {}).get("camera_model"),
        })
        if i % 10 == 0:
            log(f"[panoramax] {i + 1}/{len(route)} {name}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir
