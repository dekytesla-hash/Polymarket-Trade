from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pm5.dataset import build_training_frame, label_fixed_horizon, label_window_outcome
from pm5.features import build_features, latest_feature_row


def make_snapshots(n: int = 2000, start: float = 1_700_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = start + np.arange(n, dtype=float)
    mid = 50_000 + np.cumsum(rng.normal(0, 2, n))
    spread = np.full(n, 1.0)
    return pd.DataFrame(
        {
            "ts": ts,
            "mid": mid,
            "microprice": mid + rng.normal(0, 0.2, n),
            "spread": spread,
            "best_bid": mid - 0.5,
            "best_ask": mid + 0.5,
            "best_bid_qty": rng.uniform(1, 5, n),
            "best_ask_qty": rng.uniform(1, 5, n),
            "bid_depth": rng.uniform(50, 100, n),
            "ask_depth": rng.uniform(50, 100, n),
            "ofi_json": [json.dumps({"1": float(v), "5": float(v * 2)}) for v in rng.normal(0, 1, n)],
            "book_slope_bid": rng.uniform(1, 2, n),
            "book_slope_ask": rng.uniform(1, 2, n),
            "trade_count": rng.integers(0, 20, n),
            "buy_volume": rng.uniform(0, 3, n),
            "sell_volume": rng.uniform(0, 3, n),
            "last_trade_price": mid,
        }
    )


def test_build_features_shapes_and_columns():
    feats = build_features(make_snapshots())
    for col in [
        "momentum_60s_bps", "rv_300s_bps", "micro_dev_bps", "spread_bps", "top_imbalance",
        "depth_imbalance", "ofi_l1_60s", "trade_imbalance_5s", "tod_sin", "dow_cos",
        "seconds_left", "log_ret_since_open_bps", "open_dist_z",
    ]:
        assert col in feats.columns, col
    assert feats["seconds_left"].between(0, 300).all()
    assert np.isfinite(feats["tod_sin"]).all()


def test_cyclical_encoding_is_continuous():
    day = 86_400.0
    snaps = pd.DataFrame({"ts": [day - 1, day + 1], "mid": [100.0, 100.0]})
    feats = build_features(snaps, dropna=False)
    assert abs(feats["tod_sin"].iloc[0] - feats["tod_sin"].iloc[1]) < 1e-3
    assert abs(feats["tod_cos"].iloc[0] - feats["tod_cos"].iloc[1]) < 1e-3


def test_window_labels_match_open_close():
    snaps = make_snapshots()
    labels = label_window_outcome(snaps)
    assert set(labels["label"].unique()) <= {0, 1}
    row = labels.iloc[0]
    assert row["label"] == int(row["close_price"] >= row["open_price"])


def test_fixed_horizon_labels_align():
    snaps = make_snapshots(n=700)
    labels = label_fixed_horizon(snaps, horizon_seconds=300)
    valid = labels[labels["label_valid"]]
    assert len(valid) == 400
    mids = snaps.set_index("ts")["mid"]
    ts0 = valid["ts"].iloc[0]
    assert valid["label"].iloc[0] == int(mids[ts0 + 300] >= mids[ts0])


def test_training_frame_has_no_leakage_columns():
    frame = build_training_frame(make_snapshots(), target="window")
    assert "label" in frame
    assert "close_price" not in frame
    assert "mid" not in frame
    assert frame["seconds_left"].min() >= 10


def test_latest_feature_row_matches_batch_build():
    snaps = make_snapshots(n=400)
    row = latest_feature_row(snaps.to_dict("records"))
    batch = build_features(snaps, dropna=False).iloc[-1]
    assert row is not None
    assert abs(row["momentum_60s_bps"] - batch["momentum_60s_bps"]) < 1e-9
    assert abs(row["micro_dev_bps"] - batch["micro_dev_bps"]) < 1e-9
