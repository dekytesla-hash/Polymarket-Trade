from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pm5.config import TradingConfig
from pm5.decision import Quote, decide, kelly_fraction, realized_pnl
from pm5.ingest.binance import BookLevel, BookState, order_flow_imbalance
from pm5.ingest.polymarket import parse_book, window_start_for
from pm5.metrics import live_gate, summarize, t_stat
from pm5.model import calibration_table, evaluate_predictions, walk_forward_splits, walk_forward_train


def cfg(**kw) -> TradingConfig:
    base = TradingConfig(min_edge=0.04, cost=0.01, kelly_fraction=0.25, max_stake_usdc=5.0,
                         bankroll_usdc=500.0, max_spread=0.06, min_book_size=20.0)
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def test_kelly_fraction_bounds():
    assert kelly_fraction(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)
    assert kelly_fraction(0.4, 0.5) == 0.0
    assert 0 < kelly_fraction(0.6, 0.5) < 1
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2, abs=1e-9)


def test_no_trade_when_edge_below_threshold():
    d = decide(0.52, Quote(0.50, 0.51, 500, 500), cfg(), seconds_left=200)
    assert d.action == "no_trade"
    assert d.reason == "edge_below_threshold"


def test_buy_up_when_model_above_ask():
    d = decide(0.70, Quote(0.50, 0.52, 500, 500), cfg(), seconds_left=200)
    assert d.action == "buy_up"
    assert d.entry_price == 0.52
    assert 0 < d.size_usdc <= 5.0


def test_buy_down_when_model_below_bid():
    d = decide(0.25, Quote(0.48, 0.50, 500, 500), cfg(), seconds_left=200)
    assert d.action == "buy_down"
    assert d.entry_price == pytest.approx(0.52)


def test_stake_is_capped_and_spread_filtered():
    d = decide(0.99, Quote(0.10, 0.12, 5000, 5000), cfg(), seconds_left=200)
    assert d.size_usdc <= 5.0
    wide = decide(0.99, Quote(0.10, 0.30, 5000, 5000), cfg(), seconds_left=200)
    assert wide.reason == "spread_too_wide"


def test_expiry_and_thin_book_guards():
    assert decide(0.9, Quote(0.5, 0.52, 500, 500), cfg(), seconds_left=5).reason == "too_close_to_expiry"
    thin = decide(0.9, Quote(0.5, 0.52, 1, 1), cfg(), seconds_left=200)
    assert thin.reason == "insufficient_book_size"


def test_realized_pnl():
    assert realized_pnl("buy_up", 0.5, 10.0, 1) == pytest.approx(10.0)
    assert realized_pnl("buy_up", 0.5, 10.0, 0) == pytest.approx(-10.0)
    assert realized_pnl("buy_down", 0.25, 5.0, 0) == pytest.approx(15.0)
    assert realized_pnl("no_trade", 0.5, 0.0, 1) == 0.0


def test_order_flow_imbalance_signs():
    prev = BookState(bids=[BookLevel(100.0, 1.0)], asks=[BookLevel(101.0, 1.0)])
    more_bid = BookState(bids=[BookLevel(100.0, 3.0)], asks=[BookLevel(101.0, 1.0)])
    assert order_flow_imbalance(prev, more_bid, 1) == pytest.approx(2.0)
    bid_lifted = BookState(bids=[BookLevel(100.5, 1.0)], asks=[BookLevel(101.0, 1.0)])
    assert order_flow_imbalance(prev, bid_lifted, 1) > 0
    ask_pressure = BookState(bids=[BookLevel(100.0, 1.0)], asks=[BookLevel(100.5, 2.0)])
    assert order_flow_imbalance(prev, ask_pressure, 1) < 0


def test_microprice_leans_to_thin_side():
    book = BookState(bids=[BookLevel(100.0, 9.0)], asks=[BookLevel(101.0, 1.0)])
    assert book.mid == 100.5
    assert book.microprice > book.mid


def test_parse_book_and_window_start():
    top = parse_book("t", [{"price": "0.4", "size": "10"}, {"price": "0.45", "size": "5"}],
                     [{"price": "0.55", "size": "7"}, {"price": "0.6", "size": "3"}])
    assert top.best_bid == 0.45 and top.best_ask == 0.55
    assert top.bid_size == 5 and top.ask_size == 7
    assert top.mid == pytest.approx(0.5)
    assert window_start_for(1_789_327_812.4) == 1_789_327_800


def test_walk_forward_splits_are_ordered_and_embargoed():
    ts = np.arange(0, 10_000, dtype=float)
    splits = walk_forward_splits(ts, n_folds=4, embargo_seconds=100)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert ts[train_idx].max() + 100 <= ts[test_idx].min()
    for (_, a), (_, b) in zip(splits, splits[1:], strict=False):
        assert ts[a].max() < ts[b].min()


def _toy_frame(n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.9 * x)))
    return pd.DataFrame(
        {"ts": np.arange(n, dtype=float), "signal": x, "noise": noise,
         "label": (rng.uniform(size=n) < p).astype(int)}
    )


def test_walk_forward_train_produces_calibrated_model():
    bundle, folds, oos = walk_forward_train(_toy_frame(), n_folds=4, embargo_seconds=5)
    assert len(folds) == 4
    assert bundle.metrics["pooled_oos"]["brier"] < bundle.metrics["pooled_oos"]["brier_baseline"]
    assert bundle.metrics["pooled_oos"]["brier_skill_score"] > 0
    p = bundle.predict_proba(pd.DataFrame({"signal": [2.0, -2.0], "noise": [0.0, 0.0]}))
    assert p[0] > p[1]
    assert ((0 <= p) & (p <= 1)).all()
    cal = calibration_table(oos["label"].to_numpy(), oos["p_model"].to_numpy())
    assert (cal["observed_rate"].diff().dropna() > -0.25).all()


def test_evaluate_predictions_rewards_sharpness():
    y = np.array([1, 1, 0, 0])
    good = evaluate_predictions(y, np.array([0.9, 0.8, 0.2, 0.1]))
    bad = evaluate_predictions(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert good["brier"] < bad["brier"]
    assert good["brier_skill_score"] > 0


def test_live_gate_blocks_until_enough_evidence():
    df = pd.DataFrame(
        {
            "mode": ["paper"] * 10,
            "action": ["buy_up"] * 10,
            "outcome": [1] * 10,
            "p_model": [0.7] * 10,
            "p_market": [0.5] * 10,
            "size_usdc": [1.0] * 10,
            "pnl_usdc": [0.5] * 10,
        }
    )
    ok, reason = live_gate(df, min_trades=300, min_t_stat=2.0)
    assert not ok and "need 300" in reason
    assert summarize(df)["n_resolved_trades"] == 10
    assert t_stat(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.0)
