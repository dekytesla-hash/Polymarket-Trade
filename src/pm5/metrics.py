"""Performance tracking over logged decisions: accuracy, Brier, calibration, P&L."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pm5.model import calibration_table, evaluate_predictions
from pm5.storage import Storage


def decisions_frame(storage: Storage, mode: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM decisions"
    params: tuple[Any, ...] = ()
    if mode:
        query += " WHERE mode = ?"
        params = (mode,)
    query += " ORDER BY ts"
    rows = [dict(r) for r in storage.fetch_all(query, params)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Prediction quality over all resolved decisions + P&L over executed trades."""
    out: dict[str, Any] = {
        "n_decisions": int(len(df)),
        "n_trades": int((df["action"] != "no_trade").sum()) if len(df) else 0,
    }
    resolved = df[df["outcome"].notna()] if len(df) else df
    out["n_resolved"] = int(len(resolved))
    if len(resolved):
        y = resolved["outcome"].to_numpy(dtype=int)
        p = resolved["p_model"].to_numpy(dtype=float)
        out.update(evaluate_predictions(y, p))
        pm = resolved["p_market"].to_numpy(dtype=float)
        market_metrics = evaluate_predictions(y, np.clip(pm, 1e-6, 1 - 1e-6))
        out["market_brier"] = market_metrics["brier"]
        out["brier_vs_market"] = market_metrics["brier"] - out["brier"]

    traded = resolved[resolved["action"] != "no_trade"] if len(resolved) else resolved
    out["n_resolved_trades"] = int(len(traded))
    if len(traded):
        pnl = traded["pnl_usdc"].fillna(0.0).to_numpy(dtype=float)
        staked = traded["size_usdc"].to_numpy(dtype=float)
        out["pnl_usdc"] = float(pnl.sum())
        out["staked_usdc"] = float(staked.sum())
        out["roi"] = float(pnl.sum() / staked.sum()) if staked.sum() else 0.0
        out["win_rate"] = float((pnl > 0).mean())
        out["mean_pnl_per_trade"] = float(pnl.mean())
        out["pnl_t_stat"] = t_stat(pnl)
        out["max_drawdown_usdc"] = max_drawdown(pnl)
    return out


def t_stat(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    sd = float(np.std(pnl, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(pnl) / sd * math.sqrt(len(pnl)))


def max_drawdown(pnl: np.ndarray) -> float:
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    return float(np.min(equity - peak)) if len(equity) else 0.0


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    resolved = df[df["outcome"].notna()]
    if resolved.empty:
        return pd.DataFrame(columns=["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"])
    return calibration_table(
        resolved["outcome"].to_numpy(dtype=int), resolved["p_model"].to_numpy(dtype=float), bins=bins
    )


def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    traded = df[(df["action"] != "no_trade") & df["pnl_usdc"].notna()].sort_values("ts")
    if traded.empty:
        return pd.DataFrame(columns=["ts", "datetime", "pnl_usdc", "equity"])
    out = traded[["ts", "datetime", "pnl_usdc"]].copy()
    out["equity"] = out["pnl_usdc"].cumsum()
    return out.reset_index(drop=True)


def live_gate(df: pd.DataFrame, min_trades: int, min_t_stat: float) -> tuple[bool, str]:
    """Whether the paper record justifies enabling live execution."""
    paper = df[df["mode"] == "paper"] if "mode" in df else df
    stats = summarize(paper)
    n = stats.get("n_resolved_trades", 0)
    if n < min_trades:
        return False, f"only {n} resolved paper trades, need {min_trades}"
    t = stats.get("pnl_t_stat", 0.0)
    if t < min_t_stat:
        return False, f"paper P&L t-stat {t:.2f} below required {min_t_stat:.2f}"
    if stats.get("pnl_usdc", 0.0) <= 0:
        return False, "paper P&L is not positive"
    if stats.get("brier_vs_market", 0.0) <= 0:
        return False, "model Brier score is not better than the market's"
    return True, f"{n} paper trades, t-stat {t:.2f}, P&L {stats['pnl_usdc']:.2f} USDC"
