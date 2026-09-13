"""Dataset construction: recorded microstructure snapshots or Binance klines backfill."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pm5.features import WINDOW_SECONDS, build_features
from pm5.storage import Storage

log = logging.getLogger(__name__)


def load_recorded_snapshots(storage: Storage, since: float | None = None) -> pd.DataFrame:
    query = "SELECT * FROM btc_snapshots"
    params: tuple[float, ...] = ()
    if since is not None:
        query += " WHERE ts >= ?"
        params = (since,)
    query += " ORDER BY ts"
    rows = storage.fetch_all(query, params)
    return pd.DataFrame([dict(r) for r in rows])


def backfill_klines(
    symbol: str = "BTCUSDT",
    days: float = 7.0,
    interval: str = "1s",
    rest_base: str = "https://data-api.binance.vision",
    end_ms: int | None = None,
) -> pd.DataFrame:
    """Fetch klines and return a snapshot-shaped frame (ts, mid) for backtesting.

    Only price-derived features are available from klines; microstructure
    features require the recorded stream.
    """
    end_ms = end_ms or int(time.time() * 1000)
    step_ms = {"1s": 1000, "1m": 60_000, "5m": 300_000}[interval]
    start_ms = end_ms - int(days * 86_400_000)
    rows: list[list[float]] = []
    cursor = start_ms
    session = requests.Session()
    session.headers["User-Agent"] = "pm5/0.1"
    while cursor < end_ms:
        resp = session.get(
            f"{rest_base}/api/v3/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": min(cursor + 1000 * step_ms, end_ms),
                "limit": 1000,
            },
            timeout=20,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            cursor += 1000 * step_ms
            continue
        rows.extend(batch)
        cursor = int(batch[-1][0]) + step_ms
        log.debug("backfill at %s (%d rows)", pd.to_datetime(cursor, unit="ms"), len(rows))
    if not rows:
        return pd.DataFrame(columns=["ts", "mid"])
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    out = pd.DataFrame(
        {
            "ts": df["close_time"].astype("int64") / 1000.0,
            "mid": df["close"].astype(float),
            "buy_volume": df["taker_buy_base"].astype(float),
            "sell_volume": df["volume"].astype(float) - df["taker_buy_base"].astype(float),
            "trade_count": df["trades"].astype(int),
        }
    )
    return out.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


def label_window_outcome(
    snapshots: pd.DataFrame, window_seconds: int = WINDOW_SECONDS
) -> pd.DataFrame:
    """Per-window Up/Down label: close of window >= price at window open."""
    df = snapshots.sort_values("ts").reset_index(drop=True)
    ts = df["ts"].to_numpy(dtype=float)
    mid = df["mid"].to_numpy(dtype=float)
    window = np.floor(ts / window_seconds) * window_seconds
    frame = pd.DataFrame({"window_start": window, "ts": ts, "mid": mid})
    grouped = frame.groupby("window_start")
    opens = grouped.first()
    closes = grouped.last()
    counts = grouped.size()
    labels = pd.DataFrame(
        {
            "window_start": opens.index,
            "open_price": opens["mid"].to_numpy(),
            "close_price": closes["mid"].to_numpy(),
            "open_ts": opens["ts"].to_numpy(),
            "close_ts": closes["ts"].to_numpy(),
            "n_obs": counts.to_numpy(),
        }
    )
    labels["label"] = (labels["close_price"] >= labels["open_price"]).astype(int)
    # Drop windows with missing coverage at either edge.
    good = (labels["open_ts"] - labels["window_start"] <= 5) & (
        labels["window_start"] + window_seconds - labels["close_ts"] <= 5
    )
    return labels[good].reset_index(drop=True)


def label_fixed_horizon(
    snapshots: pd.DataFrame, horizon_seconds: int = 300, tolerance: float = 5.0
) -> pd.DataFrame:
    """Label each observation by whether price is higher ``horizon`` seconds later."""
    df = snapshots.sort_values("ts").reset_index(drop=True)
    ts = df["ts"].to_numpy(dtype=float)
    mid = df["mid"].to_numpy(dtype=float)
    idx = np.searchsorted(ts, ts + horizon_seconds, side="left")
    valid = idx < len(ts)
    idx_clipped = np.where(valid, idx, len(ts) - 1)
    future_ts = ts[idx_clipped]
    ok = valid & (np.abs(future_ts - (ts + horizon_seconds)) <= tolerance)
    return pd.DataFrame(
        {"ts": ts, "label": (mid[idx_clipped] >= mid).astype(int), "label_valid": ok}
    )


def build_training_frame(
    snapshots: pd.DataFrame,
    target: str = "window",
    window_seconds: int = WINDOW_SECONDS,
    horizon_seconds: int = 300,
    min_seconds_left: float = 10.0,
) -> pd.DataFrame:
    """Feature matrix + label, ready for walk-forward training.

    ``target="window"`` matches the tradable contract (price at window close vs
    window open); ``target="horizon"`` is the generic "higher in 5 minutes".
    """
    feats = build_features(snapshots, window_seconds=window_seconds)
    if feats.empty:
        return feats
    if target == "window":
        labels = label_window_outcome(snapshots, window_seconds)
        merged = feats.merge(labels[["window_start", "label"]], on="window_start", how="inner")
        merged = merged[merged["seconds_left"] >= min_seconds_left]
    elif target == "horizon":
        labels = label_fixed_horizon(snapshots, horizon_seconds)
        merged = feats.merge(labels, on="ts", how="inner")
        merged = merged[merged["label_valid"]].drop(columns=["label_valid"])
    else:
        raise ValueError(f"unknown target: {target}")
    return merged.dropna(subset=["label"]).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
