"""Streamlit dashboard: decisions, accuracy, calibration and simulated P&L."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pm5.config import Config
from pm5.metrics import calibration, decisions_frame, equity_curve, live_gate, summarize
from pm5.storage import Storage

st.set_page_config(page_title="pm5 - Polymarket 5m BTC", layout="wide")


@st.cache_data(ttl=15)
def load(db_path: str, mode: str | None) -> pd.DataFrame:
    with Storage(db_path) as storage:
        return decisions_frame(storage, mode=mode)


def main() -> None:
    cfg = Config.load()
    st.title("Polymarket 5-minute BTC Up/Down")

    with st.sidebar:
        db_path = st.text_input("Database", str(cfg.storage.db_path))
        mode = st.selectbox("Mode", ["all", "paper", "live"])
        st.caption("Refresh the page to reload data (15s cache).")

    if not Path(db_path).exists():
        st.warning(f"No database at {db_path}. Run `pm5 record` / `pm5 paper` first.")
        return

    df = load(db_path, None if mode == "all" else mode)
    if df.empty:
        st.info("No decisions logged yet.")
        return

    stats = summarize(df)
    cols = st.columns(6)
    cols[0].metric("Decisions", stats["n_decisions"])
    cols[1].metric("Trades", stats["n_trades"])
    cols[2].metric("Resolved", stats["n_resolved"])
    cols[3].metric("Brier", f"{stats.get('brier', float('nan')):.4f}")
    cols[4].metric("Brier vs market", f"{stats.get('brier_vs_market', 0.0):+.4f}")
    cols[5].metric("P&L (USDC)", f"{stats.get('pnl_usdc', 0.0):+.2f}")

    ok, reason = live_gate(df, cfg.trading.live_min_trades, cfg.trading.live_min_t_stat)
    (st.success if ok else st.warning)(f"Live gate: {'PASS' if ok else 'BLOCKED'} — {reason}")

    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative P&L")
        eq = equity_curve(df)
        if eq.empty:
            st.caption("No settled trades yet.")
        else:
            st.line_chart(eq.set_index("datetime")["equity"])
    with right:
        st.subheader("Calibration")
        cal = calibration(df)
        if cal.empty:
            st.caption("No resolved decisions yet.")
        else:
            chart = cal.set_index("mean_predicted")[["observed_rate"]].copy()
            chart["perfect"] = chart.index
            st.line_chart(chart)
            st.dataframe(cal, use_container_width=True)

    st.subheader("Edge vs outcome")
    resolved = df[df["outcome"].notna()]
    if not resolved.empty:
        st.scatter_chart(resolved, x="p_market", y="p_model", color="outcome")

    st.subheader("Decision log")
    show = df.sort_values("ts", ascending=False).head(500)
    columns = [
        "datetime", "slug", "mode", "p_model", "p_market", "edge", "action", "reason",
        "size_usdc", "entry_price", "outcome", "pnl_usdc",
    ]
    st.dataframe(show[[c for c in columns if c in show.columns]], use_container_width=True)

    with st.expander("Model metrics"):
        metrics_path = cfg.model.model_dir / "model.metrics.json"
        if metrics_path.exists():
            st.json(json.loads(metrics_path.read_text()))
        else:
            st.caption("Train a model to see walk-forward metrics here.")

    with st.expander("Summary JSON"):
        st.json(stats)


main()
