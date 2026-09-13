"""SQLite storage plus Parquet export for all recorded data and decisions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS btc_snapshots (
    ts REAL PRIMARY KEY,
    mid REAL NOT NULL,
    microprice REAL NOT NULL,
    spread REAL NOT NULL,
    best_bid REAL NOT NULL,
    best_ask REAL NOT NULL,
    best_bid_qty REAL NOT NULL,
    best_ask_qty REAL NOT NULL,
    bid_depth REAL NOT NULL,
    ask_depth REAL NOT NULL,
    ofi_json TEXT NOT NULL,
    book_slope_bid REAL,
    book_slope_ask REAL,
    trade_count INTEGER NOT NULL DEFAULT 0,
    buy_volume REAL NOT NULL DEFAULT 0,
    sell_volume REAL NOT NULL DEFAULT 0,
    last_trade_price REAL
);

CREATE TABLE IF NOT EXISTS pm_markets (
    slug TEXT PRIMARY KEY,
    condition_id TEXT,
    question TEXT,
    up_token_id TEXT,
    down_token_id TEXT,
    window_start REAL,
    window_end REAL,
    resolved_outcome INTEGER,
    resolved_at REAL,
    outcome_source TEXT
);

CREATE TABLE IF NOT EXISTS pm_quotes (
    ts REAL NOT NULL,
    slug TEXT NOT NULL,
    token_id TEXT NOT NULL,
    side TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    bid_size REAL,
    ask_size REAL,
    mid REAL,
    PRIMARY KEY (ts, token_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    slug TEXT NOT NULL,
    window_start REAL,
    window_end REAL,
    mode TEXT NOT NULL,
    features_json TEXT NOT NULL,
    p_model REAL NOT NULL,
    p_market REAL NOT NULL,
    edge REAL NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    size_usdc REAL NOT NULL,
    entry_price REAL,
    token_id TEXT,
    model_version TEXT,
    outcome INTEGER,
    pnl_usdc REAL,
    resolved_at REAL,
    order_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_slug ON decisions(slug);
CREATE INDEX IF NOT EXISTS idx_quotes_slug ON pm_quotes(slug);
CREATE INDEX IF NOT EXISTS idx_snap_ts ON btc_snapshots(ts);
"""


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def insert_btc_snapshot(self, snap: dict[str, Any]) -> None:
        row = dict(snap)
        row["ofi_json"] = json.dumps(row.pop("ofi", {}))
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(
            f"INSERT OR REPLACE INTO btc_snapshots ({cols}) VALUES ({marks})", list(row.values())
        )
        self.conn.commit()

    def upsert_market(self, market: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO pm_markets (slug, condition_id, question, up_token_id, down_token_id,
                                    window_start, window_end)
            VALUES (:slug, :condition_id, :question, :up_token_id, :down_token_id,
                    :window_start, :window_end)
            ON CONFLICT(slug) DO UPDATE SET
                condition_id=excluded.condition_id,
                question=excluded.question,
                up_token_id=excluded.up_token_id,
                down_token_id=excluded.down_token_id,
                window_start=excluded.window_start,
                window_end=excluded.window_end
            """,
            market,
        )
        self.conn.commit()

    def insert_quote(self, quote: dict[str, Any]) -> None:
        cols = ", ".join(quote)
        marks = ", ".join("?" for _ in quote)
        self.conn.execute(
            f"INSERT OR REPLACE INTO pm_quotes ({cols}) VALUES ({marks})", list(quote.values())
        )
        self.conn.commit()

    def insert_decision(self, decision: dict[str, Any]) -> int:
        row = dict(decision)
        row["features_json"] = json.dumps(row.pop("features", {}))
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.conn.execute(f"INSERT INTO decisions ({cols}) VALUES ({marks})", list(row.values()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def set_market_outcome(
        self, slug: str, outcome: int, resolved_at: float, source: str = "gamma"
    ) -> None:
        self.conn.execute("INSERT OR IGNORE INTO pm_markets (slug) VALUES (?)", (slug,))
        self.conn.execute(
            "UPDATE pm_markets SET resolved_outcome=?, resolved_at=?, outcome_source=? WHERE slug=?",
            (outcome, resolved_at, source, slug),
        )
        self.conn.commit()

    def settle_decision(self, decision_id: int, outcome: int, pnl: float, resolved_at: float) -> None:
        self.conn.execute(
            "UPDATE decisions SET outcome=?, pnl_usdc=?, resolved_at=? WHERE id=?",
            (outcome, pnl, resolved_at, decision_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------- reads
    def unresolved_decisions(self, before_ts: float) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM decisions WHERE outcome IS NULL AND window_end IS NOT NULL "
                "AND window_end < ? ORDER BY window_end",
                (before_ts,),
            )
        )

    def market(self, slug: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pm_markets WHERE slug=?", (slug,)).fetchone()

    def fetch_all(self, query: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(query, params))

    # ------------------------------------------------------------------ export
    def export_parquet(self, out_dir: str | Path, tables: Iterable[str] | None = None) -> list[Path]:
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tables = tables or ("btc_snapshots", "pm_markets", "pm_quotes", "decisions")
        written: list[Path] = []
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
            path = out / f"{table}.parquet"
            df.to_parquet(path, index=False)
            written.append(path)
        return written
