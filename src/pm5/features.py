"""Feature engineering.

A single vectorised implementation is used for both training and inference: the
live engine keeps a rolling buffer of 1-second snapshots, converts it to the
same DataFrame shape used at training time, and takes the last row. That
guarantees train/serve consistency.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

WINDOW_SECONDS = 300

# Rolling windows (seconds) used for flow/volatility aggregates.
FLOW_WINDOWS = (5, 60, 300)
PRICE_WINDOWS = (60, 300, 900)


def _rolling_sum(series: pd.Series, window_s: int, ts: pd.Series) -> pd.Series:
    idx = pd.to_datetime(ts, unit="s")
    out = series.copy()
    out.index = idx
    return out.rolling(f"{window_s}s").sum().to_numpy()


def _rolling_std(series: pd.Series, window_s: int, ts: pd.Series) -> pd.Series:
    idx = pd.to_datetime(ts, unit="s")
    out = series.copy()
    out.index = idx
    return out.rolling(f"{window_s}s").std().to_numpy()


def _value_lagged(ts: np.ndarray, values: np.ndarray, lag_s: float) -> np.ndarray:
    """Value of ``values`` as of ``lag_s`` seconds before each timestamp."""
    positions = np.searchsorted(ts, ts - lag_s, side="right") - 1
    positions = np.clip(positions, 0, len(values) - 1)
    out = values[positions]
    return np.where(positions < 0, np.nan, out)


def expand_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the stored ``ofi_json`` column into ofi_l{level} columns."""
    if "ofi_json" not in df.columns:
        return df
    parsed = df["ofi_json"].apply(lambda v: json.loads(v) if isinstance(v, str) else (v or {}))
    levels = sorted({int(k) for d in parsed for k in d}, key=int)
    for lvl in levels:
        df[f"ofi_l{lvl}"] = parsed.apply(lambda d, lvl=lvl: float(d.get(str(lvl), d.get(lvl, 0.0))))
    return df.drop(columns=["ofi_json"])


def build_features(
    snapshots: pd.DataFrame,
    window_seconds: int = WINDOW_SECONDS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix from a time-ordered frame of BTC snapshots.

    Required columns: ts, mid. Optional microstructure columns are used when
    present so that klines-only backfills work with a reduced feature set.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _build_features(snapshots, window_seconds, dropna)


def _build_features(snapshots: pd.DataFrame, window_seconds: int, dropna: bool) -> pd.DataFrame:
    df = snapshots.sort_values("ts").reset_index(drop=True).copy()
    df = expand_ofi(df)
    ts = df["ts"].to_numpy(dtype=float)
    mid = df["mid"].to_numpy(dtype=float)
    out = pd.DataFrame({"ts": ts})

    # --- price / momentum / volatility -----------------------------------
    log_mid = np.log(np.where(mid > 0, mid, np.nan))
    out["log_mid"] = log_mid
    ret_1 = np.diff(log_mid, prepend=log_mid[0])
    for w in PRICE_WINDOWS:
        past = _value_lagged(ts, log_mid, w)
        out[f"momentum_{w}s_bps"] = (log_mid - past) * 1e4
        out[f"rv_{w}s_bps"] = _rolling_std(pd.Series(ret_1), w, df["ts"]) * 1e4 * math.sqrt(w)
    out["ret_1s_bps"] = ret_1 * 1e4

    # --- position inside the 5-minute market window ----------------------
    window_start = np.floor(ts / window_seconds) * window_seconds
    open_positions = np.searchsorted(ts, window_start, side="left")
    open_positions = np.clip(open_positions, 0, len(mid) - 1)
    open_price = mid[open_positions]
    stale_open = ts[open_positions] > window_start + 5  # first tick of window is too late
    out["window_start"] = window_start
    out["seconds_into_window"] = ts - window_start
    out["seconds_left"] = window_start + window_seconds - ts
    out["log_ret_since_open_bps"] = np.where(stale_open, np.nan, (log_mid - np.log(open_price)) * 1e4)

    # Distance from the open scaled by the volatility of the remaining horizon,
    # i.e. roughly the z-score that decides the Up/Down outcome.
    rv_per_s = out["rv_60s_bps"].to_numpy() / math.sqrt(60.0)
    horizon_vol = rv_per_s * np.sqrt(np.maximum(out["seconds_left"].to_numpy(), 1.0))
    out["open_dist_z"] = out["log_ret_since_open_bps"].to_numpy() / np.where(
        horizon_vol > 0, horizon_vol, np.nan
    )

    # --- microstructure ---------------------------------------------------
    if "microprice" in df:
        micro = df["microprice"].to_numpy(dtype=float)
        out["micro_dev_bps"] = np.where(mid > 0, (micro - mid) / mid * 1e4, np.nan)
    if "spread" in df:
        out["spread_bps"] = np.where(mid > 0, df["spread"].to_numpy() / mid * 1e4, np.nan)
    if {"best_bid_qty", "best_ask_qty"} <= set(df.columns):
        bq = df["best_bid_qty"].to_numpy(dtype=float)
        aq = df["best_ask_qty"].to_numpy(dtype=float)
        out["top_imbalance"] = np.where(bq + aq > 0, (bq - aq) / (bq + aq), 0.0)
    if {"bid_depth", "ask_depth"} <= set(df.columns):
        bd = df["bid_depth"].to_numpy(dtype=float)
        ad = df["ask_depth"].to_numpy(dtype=float)
        out["depth_imbalance"] = np.where(bd + ad > 0, (bd - ad) / (bd + ad), 0.0)
        out["log_depth"] = np.log1p(bd + ad)
    if {"book_slope_bid", "book_slope_ask"} <= set(df.columns):
        sb = df["book_slope_bid"].to_numpy(dtype=float)
        sa = df["book_slope_ask"].to_numpy(dtype=float)
        out["slope_imbalance"] = np.where(sb + sa > 0, (sb - sa) / (sb + sa), 0.0)
        out["log_slope_total"] = np.log1p(np.maximum(sb + sa, 0.0))

    # --- order flow imbalance across depth levels -------------------------
    ofi_cols = [c for c in df.columns if c.startswith("ofi_l")]
    scale = df.get("bid_depth", pd.Series(np.ones(len(df)))) + df.get(
        "ask_depth", pd.Series(np.ones(len(df)))
    )
    scale = scale.replace(0, np.nan).to_numpy(dtype=float)
    for col in ofi_cols:
        for w in FLOW_WINDOWS:
            out[f"{col}_{w}s"] = _rolling_sum(df[col].astype(float), w, df["ts"]) / scale

    # --- trade flow -------------------------------------------------------
    if {"buy_volume", "sell_volume"} <= set(df.columns):
        for w in FLOW_WINDOWS:
            buy = _rolling_sum(df["buy_volume"].astype(float), w, df["ts"])
            sell = _rolling_sum(df["sell_volume"].astype(float), w, df["ts"])
            total = buy + sell
            out[f"trade_imbalance_{w}s"] = np.where(total > 0, (buy - sell) / total, 0.0)
            out[f"log_volume_{w}s"] = np.log1p(total)

    # --- calendar (cyclical) ---------------------------------------------
    dt = pd.to_datetime(ts, unit="s", utc=True)
    seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    frac_day = seconds_of_day / 86400.0
    out["tod_sin"] = np.sin(2 * np.pi * frac_day)
    out["tod_cos"] = np.cos(2 * np.pi * frac_day)
    frac_week = (dt.dayofweek + frac_day) / 7.0
    out["dow_sin"] = np.sin(2 * np.pi * frac_week)
    out["dow_cos"] = np.cos(2 * np.pi * frac_week)

    out = out.replace([np.inf, -np.inf], np.nan)
    if dropna:
        out = out.dropna(subset=[c for c in out.columns if c.startswith("momentum_")])
    return out.reset_index(drop=True)


NON_FEATURE_COLUMNS = {"ts", "window_start", "log_mid", "label", "mid", "outcome"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def latest_feature_row(
    snapshots: list[dict[str, Any]], window_seconds: int = WINDOW_SECONDS
) -> dict[str, float] | None:
    """Feature vector for the most recent snapshot in a live rolling buffer."""
    if len(snapshots) < 2:
        return None
    df = pd.DataFrame(snapshots)
    feats = build_features(df, window_seconds=window_seconds, dropna=False)
    if feats.empty:
        return None
    row = feats.iloc[-1]
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
