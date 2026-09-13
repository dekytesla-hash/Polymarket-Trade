"""Real-time engine: record data, make decisions, log them, and settle outcomes.

Runs in ``paper`` mode by default. Live execution is gated on the paper-trading
record (see :func:`pm5.metrics.live_gate`) and on an explicit configuration flag.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from pm5.config import Config
from pm5.decision import Decision, Quote, decide, realized_pnl
from pm5.features import latest_feature_row
from pm5.ingest.binance import BinanceOrderBookStream
from pm5.ingest.polymarket import ClobMarketStream, GammaClient, Market, MarketTracker
from pm5.metrics import decisions_frame, live_gate
from pm5.model import ModelBundle
from pm5.storage import Storage

log = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        cfg: Config,
        storage: Storage,
        model: ModelBundle | None = None,
        record_only: bool = False,
        executor: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.model = model
        self.record_only = record_only or model is None
        self.executor = executor
        self.binance = BinanceOrderBookStream(
            symbol=cfg.binance.symbol,
            ws_base=cfg.binance.ws_base,
            depth_levels=cfg.binance.depth_levels,
            depth_update_ms=cfg.binance.depth_update_ms,
            ofi_levels=tuple(cfg.features.ofi_levels),
            trade_buffer_s=max(cfg.features.momentum_windows_s) * 2,
        )
        self.gamma = GammaClient(
            base=cfg.polymarket.gamma_base,
            slug_prefix=cfg.polymarket.slug_prefix,
            window_seconds=cfg.polymarket.window_seconds,
        )
        self.clob = ClobMarketStream(url=cfg.polymarket.clob_ws)
        self.tracker = MarketTracker(self.gamma, self.clob)
        self.tracker.on_new_market(self._on_new_market)
        self.snapshots: deque[dict[str, Any]] = deque(
            maxlen=int(max(cfg.features.momentum_windows_s) / cfg.storage.snapshot_interval_s) + 60
        )
        self.traded_markets: set[str] = set()
        # If Gamma has not published a resolution this long after a window closes,
        # fall back to Binance klines (an approximation of the Chainlink source).
        self.fallback_after_s = 1800.0
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------ setup
    def _on_new_market(self, market: Market) -> None:
        self.storage.upsert_market(market.as_row())

    async def run(self, duration_s: float | None = None) -> None:
        self._tasks = [
            self.binance.start(),
            self.clob.start(),
            self.tracker.start(),
            asyncio.create_task(self._snapshot_loop(), name="snapshot-loop"),
            asyncio.create_task(self._decision_loop(), name="decision-loop"),
            asyncio.create_task(self._settlement_loop(), name="settlement-loop"),
        ]
        try:
            if duration_s:
                await asyncio.sleep(duration_s)
            else:
                await asyncio.gather(*self._tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.gamma.close()

    # ------------------------------------------------------------------ loops
    async def _snapshot_loop(self) -> None:
        interval = self.cfg.storage.snapshot_interval_s
        while True:
            snap = self.binance.take_snapshot()
            if snap:
                self.snapshots.append(snap)
                try:
                    self.storage.insert_btc_snapshot(snap)
                except Exception as exc:  # noqa: BLE001 - never kill the loop on a write error
                    log.warning("snapshot write failed: %s", exc)
            self._record_pm_quote()
            await asyncio.sleep(interval)

    def _record_pm_quote(self) -> None:
        market = self.tracker.market
        if market is None:
            return
        for side, token_id in (("up", market.up_token_id), ("down", market.down_token_id)):
            book = self.clob.books.get(token_id)
            if book is None:
                continue
            with contextlib.suppress(Exception):
                self.storage.insert_quote(
                    {
                        "ts": time.time(),
                        "slug": market.slug,
                        "token_id": token_id,
                        "side": side,
                        "best_bid": book.best_bid,
                        "best_ask": book.best_ask,
                        "bid_size": book.bid_size,
                        "ask_size": book.ask_size,
                        "mid": book.mid,
                    }
                )

    async def _decision_loop(self) -> None:
        while True:
            try:
                await self.step()
            except Exception as exc:  # noqa: BLE001 - keep trading loop alive
                log.exception("decision step failed: %s", exc)
            await asyncio.sleep(1.0)

    async def step(self) -> Decision | None:
        if self.record_only or self.model is None:
            return None
        market = self.tracker.market
        if market is None or market.slug in self.traded_markets:
            return None
        now = time.time()
        into_window = now - market.window_start
        if into_window < self.cfg.polymarket.no_trade_head_seconds:
            return None
        if market.seconds_left <= self.cfg.polymarket.no_trade_tail_seconds:
            return None
        # Feature building and inference are CPU-bound; keep them off the event
        # loop so the websocket readers never fall behind.
        feats = await asyncio.to_thread(
            latest_feature_row, list(self.snapshots), self.cfg.polymarket.window_seconds
        )
        if feats is None:
            return None
        missing = [f for f in self.model.features if feats.get(f) is None]
        if missing:
            log.debug("skipping decision, missing features: %s", missing[:5])
            return None

        X = pd.DataFrame([feats])
        p_model = float((await asyncio.to_thread(self.model.predict_proba, X))[0])

        up_book = self.clob.books.get(market.up_token_id)
        if up_book is None:
            return None
        quote = Quote(up_book.best_bid, up_book.best_ask, up_book.bid_size, up_book.ask_size)
        trading = self.cfg.trading
        max_stake = trading.live_max_stake_usdc if self.cfg.mode == "live" else trading.max_stake_usdc
        dec = decide(
            p_model,
            quote,
            self.cfg.trading,
            seconds_left=market.seconds_left,
            no_trade_tail_seconds=self.cfg.polymarket.no_trade_tail_seconds,
            max_stake_override=max_stake,
        )
        order_id = None
        if dec.action != "no_trade":
            self.traded_markets.add(market.slug)
            if self.cfg.mode == "live" and self.executor is not None:
                order_id = await self.executor.place(market, dec)
        self._log_decision(market, dec, feats, order_id)
        return dec

    def _log_decision(
        self, market: Market, dec: Decision, feats: dict[str, float], order_id: str | None
    ) -> int:
        token_id = (
            market.up_token_id
            if dec.action == "buy_up"
            else market.down_token_id
            if dec.action == "buy_down"
            else None
        )
        row = {
            "ts": time.time(),
            "slug": market.slug,
            "window_start": market.window_start,
            "window_end": market.window_end,
            "mode": self.cfg.mode,
            "features": feats,
            "p_model": dec.p_model,
            "p_market": dec.p_market,
            "edge": dec.edge,
            "action": dec.action,
            "reason": dec.reason,
            "size_usdc": dec.size_usdc,
            "entry_price": dec.entry_price,
            "token_id": token_id,
            "model_version": self.model.version if self.model else None,
            "order_id": order_id,
        }
        decision_id = self.storage.insert_decision(row)
        log.info(
            "%s %s p_model=%.3f p_mkt=%.3f edge=%+.3f size=%.2f (%s)",
            market.slug, dec.action, dec.p_model, dec.p_market, dec.edge, dec.size_usdc, dec.reason,
        )
        return decision_id

    async def _settlement_loop(self) -> None:
        while True:
            try:
                await self.settle_pending()
            except Exception as exc:  # noqa: BLE001
                log.warning("settlement failed: %s", exc)
            await asyncio.sleep(30.0)

    async def settle_pending(self) -> int:
        """Fetch outcomes for resolved markets and settle logged decisions."""
        now = time.time()
        pending = self.storage.unresolved_decisions(now - 30)
        settled = 0
        outcomes: dict[str, int | None] = {}
        for row in pending:
            slug = row["slug"]
            if slug not in outcomes:
                outcomes[slug] = await self._resolve_market(slug, float(row["window_end"]), now)
            outcome = outcomes[slug]
            if outcome is None:
                continue
            pnl = realized_pnl(
                row["action"], row["entry_price"] or 0.0, row["size_usdc"] or 0.0, outcome
            )
            self.storage.settle_decision(int(row["id"]), outcome, pnl, time.time())
            settled += 1
        if settled:
            log.info("settled %d decisions", settled)
        return settled

    async def _resolve_market(self, slug: str, window_end: float, now: float) -> int | None:
        cached = self.storage.market(slug)
        if cached is not None and cached["resolved_outcome"] is not None:
            return int(cached["resolved_outcome"])
        outcome = await self.gamma.resolution(slug)
        if outcome is not None:
            self.storage.set_market_outcome(slug, outcome, now, source="gamma")
            return outcome
        if now - window_end < self.fallback_after_s:
            return None
        outcome = await self._binance_outcome(window_end - self.cfg.polymarket.window_seconds, window_end)
        if outcome is not None:
            log.warning("resolved %s from Binance klines (Gamma still unresolved)", slug)
            self.storage.set_market_outcome(slug, outcome, now, source="binance_fallback")
        return outcome

    async def _binance_outcome(self, window_start: float, window_end: float) -> int | None:
        """Approximate the Up/Down outcome from Binance 1s klines over the window."""
        params = {
            "symbol": self.cfg.binance.symbol.upper(),
            "interval": "1s",
            "startTime": int(window_start * 1000),
            "endTime": int(window_end * 1000),
            "limit": 1000,
        }
        url = f"{self.cfg.binance.rest_base}/api/v3/klines"
        try:
            async with aiohttp.ClientSession() as session, session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                klines = await resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("binance fallback resolution failed: %s", exc)
            return None
        if not klines:
            return None
        return int(float(klines[-1][4]) >= float(klines[0][1]))


def load_model(cfg: Config, path: str | Path | None = None) -> ModelBundle | None:
    model_path = Path(path) if path else cfg.model.model_dir / "model.pkl"
    if not model_path.exists():
        log.warning("no model found at %s; running in record-only mode", model_path)
        return None
    return ModelBundle.load(model_path)


def assert_live_allowed(cfg: Config, storage: Storage) -> None:
    """Raise unless the paper-trading record clears the live-trading gate."""
    df = decisions_frame(storage)
    ok, reason = live_gate(df, cfg.trading.live_min_trades, cfg.trading.live_min_t_stat)
    if not ok:
        raise RuntimeError(f"live trading blocked: {reason}")
    log.warning("live trading gate passed: %s", reason)
