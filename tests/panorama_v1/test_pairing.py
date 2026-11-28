from pairing import merge_retrieval_pairs, sequence_pairs


def test_sequence_pairs_include_forward_neighbors_only():
    names = ["000.jpg", "001.jpg", "002.jpg", "003.jpg"]
    pairs = sequence_pairs(names, sequence_window=2)
    assert ("000.jpg", "001.jpg") in pairs
    assert ("000.jpg", "002.jpg") in pairs
    assert ("001.jpg", "003.jpg") in pairs
    assert ("000.jpg", "003.jpg") not in pairs


def test_global_retrieval_pairs_are_merged_when_available():
    base = {("a.jpg", "b.jpg")}
    pairs, used, error = merge_retrieval_pairs(
        base, lambda: [("a.jpg", "c.jpg"), ("b.jpg", "b.jpg")]
    )
    assert used is True
    assert error is None
    assert pairs == {("a.jpg", "b.jpg"), ("a.jpg", "c.jpg")}


def test_sequential_pairs_survive_global_retrieval_timeout():
    base = {("a.jpg", "b.jpg"), ("b.jpg", "c.jpg")}

    def failing_retriever():
        raise TimeoutError("NetVLAD checkpoint download timed out")

    pairs, used, error = merge_retrieval_pairs(base, failing_retriever)
    assert used is False
    assert "TimeoutError" in error
    assert "timed out" in error
    assert pairs == base
