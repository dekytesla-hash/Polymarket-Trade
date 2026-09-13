from __future__ import annotations

import time

import pytest

from pm5.config import Config
from pm5.decision import Decision
from pm5.engine import Engine, assert_live_allowed
from pm5.ingest.polymarket import Market
from pm5.storage import Storage


@pytest.fixture()
def storage(tmp_path):
    with Storage(tmp_path / "test.db") as st:
        yield st


def _market(slug: str = "btc-updown-5m-1789327800") -> Market:
    start = 1_789_327_800.0
    return Market(slug, "0xcond", "Bitcoin Up or Down", "up-token", "down-token", start, start + 300)


def test_storage_roundtrip_and_settlement(storage):
    storage.upsert_market(_market().as_row())
    storage.insert_btc_snapshot(
        {
            "ts": 1.0, "mid": 100.0, "microprice": 100.1, "spread": 0.2, "best_bid": 99.9,
            "best_ask": 100.1, "best_bid_qty": 1.0, "best_ask_qty": 2.0, "bid_depth": 10.0,
            "ask_depth": 11.0, "ofi": {"1": 0.5}, "book_slope_bid": 1.0, "book_slope_ask": 1.1,
            "trade_count": 3, "buy_volume": 1.0, "sell_volume": 0.5, "last_trade_price": 100.0,
        }
    )
    did = storage.insert_decision(
        {
            "ts": 1_789_327_850.0, "slug": _market().slug, "window_start": 1_789_327_800.0,
            "window_end": 1_789_328_100.0, "mode": "paper", "features": {"a": 1.0},
            "p_model": 0.6, "p_market": 0.5, "edge": 0.1, "action": "buy_up", "reason": "trade",
            "size_usdc": 2.0, "entry_price": 0.5, "token_id": "up-token", "model_version": "v1",
            "order_id": None,
        }
    )
    assert did > 0
    assert len(storage.unresolved_decisions(time.time())) == 1
    storage.settle_decision(did, 1, 2.0, time.time())
    assert storage.unresolved_decisions(time.time()) == []
    row = storage.fetch_all("SELECT * FROM decisions")[0]
    assert row["outcome"] == 1 and row["pnl_usdc"] == 2.0
    assert '"a": 1.0' in row["features_json"]


def test_parquet_export(storage, tmp_path):
    storage.upsert_market(_market().as_row())
    paths = storage.export_parquet(tmp_path / "pq")
    assert {p.name for p in paths} == {
        "btc_snapshots.parquet", "pm_markets.parquet", "pm_quotes.parquet", "decisions.parquet"
    }


def test_engine_logs_decision_and_settles(storage, monkeypatch):
    cfg = Config()
    engine = Engine(cfg, storage, model=None, record_only=True)
    market = _market()
    dec = Decision("buy_up", "trade", 0.62, 0.5, 0.12, size_usdc=3.0, entry_price=0.52)
    did = engine._log_decision(market, dec, {"momentum_60s_bps": 1.5}, None)

    async def fake_resolution(slug: str) -> int:
        assert slug == market.slug
        return 1

    monkeypatch.setattr(engine.gamma, "resolution", fake_resolution)
    import asyncio

    settled = asyncio.run(engine.settle_pending())
    assert settled == 1
    row = storage.fetch_all("SELECT * FROM decisions WHERE id=?", (did,))[0]
    assert row["outcome"] == 1
    assert row["pnl_usdc"] == pytest.approx(3.0 / 0.52 - 3.0)
    assert storage.market(market.slug)["resolved_outcome"] == 1


def test_live_gate_blocks_without_paper_record(storage):
    cfg = Config()
    with pytest.raises(RuntimeError, match="live trading blocked"):
        assert_live_allowed(cfg, storage)


def test_config_loads_yaml_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "mode: paper\ntrading:\n  min_edge: 0.09\n  max_stake_usdc: 2.5\n"
        "storage:\n  db_path: /tmp/x.db\nfeatures:\n  ofi_levels: [1, 3]\n"
    )
    cfg = Config.load(path)
    assert cfg.trading.min_edge == 0.09
    assert cfg.trading.max_stake_usdc == 2.5
    assert str(cfg.storage.db_path) == "/tmp/x.db"
    assert tuple(cfg.features.ofi_levels) == (1, 3)
