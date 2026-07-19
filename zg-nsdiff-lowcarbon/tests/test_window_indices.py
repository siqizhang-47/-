import numpy as np

from src.data.window_index import build_split_indices, valid_window_starts


def test_windows_do_not_cross_split():
    N, L, H = 100, 10, 5
    ok = np.ones(N, dtype=bool)
    idx = build_split_indices(N, 0.7, 0.1, L, H, ok, ok)
    train_end, val_end = idx["train_end"], idx["val_end"]
    for split, (s, e) in [("train", (0, train_end)), ("val", (train_end, val_end)),
                          ("test", (val_end, N))]:
        starts = idx[f"{split}_starts"]
        assert (starts >= s).all()
        assert (starts + L + H <= e).all()


def test_missing_window_dropped():
    N, L, H = 60, 10, 5
    target_ok = np.ones(N, dtype=bool)
    weather_ok = np.ones(N, dtype=bool)
    target_ok[30] = False  # one NaN row
    starts_all, starts_valid = valid_window_starts(0, N, L, H, target_ok, weather_ok)
    total = L + H
    dropped = set(starts_all) - set(starts_valid)
    # every window covering row 30 must be dropped
    expected = {s for s in starts_all if s <= 30 < s + total}
    assert dropped == expected


def test_window_counts_match_formula():
    N, L, H = 500, 168, 24
    ok = np.ones(N, dtype=bool)
    starts_all, starts_valid = valid_window_starts(0, N, L, H, ok, ok)
    assert len(starts_all) == N - (L + H) + 1
    assert np.array_equal(starts_all, starts_valid)
