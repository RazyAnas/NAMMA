"""Robust image-pair construction for panorama structure-from-motion."""


def sequence_pairs(names, sequence_window):
    """Pair each ordered view with nearby views from the same capture route."""
    window = max(0, int(sequence_window))
    ordered = list(names)
    return {
        tuple(sorted((ordered[i], ordered[j])))
        for i in range(len(ordered))
        for j in range(i + 1, min(len(ordered), i + window + 1))
    }


def merge_retrieval_pairs(base_pairs, retrieval_loader):
    """Add global-retrieval pairs when available; preserve sequential pairs on failure."""
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
