"""Yaw-aware and route-aware image-pair construction for panorama SfM."""
from __future__ import annotations
import re

_VIEW_RE = re.compile(r"^p(\d+)_v(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def parse_view_name(name: str) -> tuple[int, int] | None:
    m = _VIEW_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def sequence_pairs(names, sequence_window: int = 18):
    """Pair perspective views by angular overlap across nearby panoramas on the route."""
    ordered = sorted(names)
    parsed = [(name, parse_view_name(name)) for name in ordered]
    if not all(item[1] is not None for item in parsed):
        window = max(0, int(sequence_window))
        return {
            tuple(sorted((ordered[i], ordered[j])))
            for i in range(len(ordered))
            for j in range(i + 1, min(len(ordered), i + window + 1))
        }

    by_pano: dict[int, dict[int, str]] = {}
    for name, (pi, vi) in parsed:
        by_pano.setdefault(pi, {})[vi] = name

    pano_ids = sorted(by_pano.keys())
    max_views = max(len(v) for v in by_pano.values())
    pano_reach = max(4, min(10, int(sequence_window) // 2))
    pairs: set[tuple[str, str]] = set()

    for idx, pi in enumerate(pano_ids):
        views_i = by_pano[pi]
        for vi, name_a in views_i.items():
            for dv in (-1, 1):
                vj = (vi + dv) % max_views
                if vj in views_i and name_a != views_i[vj]:
                    pairs.add(tuple(sorted((name_a, views_i[vj]))))

        for next_idx in range(idx + 1, min(len(pano_ids), idx + pano_reach + 1)):
            pj = pano_ids[next_idx]
            views_j = by_pano[pj]
            yaw_tol = 2 if (next_idx - idx) <= 3 else 1
            for vi, name_a in views_i.items():
                for dv in range(-yaw_tol, yaw_tol + 1):
                    vj = (vi + dv) % max_views
                    if vj in views_j:
                        pairs.add(tuple(sorted((name_a, views_j[vj]))))
    return pairs


def merge_retrieval_pairs(base_pairs, retrieval_loader):
    stable_pairs = set(base_pairs)
    try:
        retrieved = set()
        for pair in retrieval_loader():
            if len(pair) != 2:
                continue
            first, second = str(pair[0]), str(pair[1])
            if first != second:
                retrieved.add(tuple(sorted((first, second))))
        return stable_pairs | retrieved, True, None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return stable_pairs, False, message
