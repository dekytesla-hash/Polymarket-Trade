"""Real-time Binance order book / trade ingestion.

Maintains an L2 book from the partial-depth stream and a rolling trade tape,
and exposes point-in-time snapshots (including multi-level OFI) for features.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import websockets

log = logging.getLogger(__name__)


@dataclass
class BookLevel:
    price: float
    qty: float


@dataclass
class Trade:
    ts: float
    price: float
    qty: float
    buyer_maker: bool


@dataclass
class BookState:
    ts: float = 0.0
    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def microprice(self) -> float:
        """Size-weighted price: (Pb*Qa + Pa*Qb) / (Qa + Qb)."""
        if not self.bids or not self.asks:
            return 0.0
        qb, qa = self.bids[0].qty, self.asks[0].qty
        if qb + qa <= 0:
            return self.mid
        return (self.best_bid * qa + self.best_ask * qb) / (qa + qb)

    def depth(self, levels: int, side: str) -> float:
        book = self.bids if side == "bid" else self.asks
        return sum(lvl.qty for lvl in book[:levels])

    def slope(self, levels: int, side: str) -> float:
        """Cumulative size per unit of price distance from mid (liquidity density)."""
        book = self.bids if side == "bid" else self.asks
        mid = self.mid
        if not book or mid <= 0:
            return 0.0
        cum = 0.0
        num = 0.0
        for lvl in book[:levels]:
            cum += lvl.qty
            dist = abs(lvl.price - mid) / mid
            if dist > 0:
                num += cum / dist
        return num / max(1, min(levels, len(book)))


class BinanceOrderBookStream:
    """Consumes ``<symbol>@depth<N>@<ms>ms`` and ``<symbol>@aggTrade``."""

    def __init__(
        self,
        symbol: str = "btcusdt",
        ws_base: str = "wss://data-stream.binance.vision/stream",
        depth_levels: int = 20,
        depth_update_ms: int = 100,
        trade_buffer_s: float = 1800.0,
        ofi_levels: tuple[int, ...] = (1, 5, 10, 20),
    ) -> None:
        self.symbol = symbol.lower()
        self.ws_base = ws_base
        self.depth_levels = depth_levels
        self.depth_update_ms = depth_update_ms
        self.ofi_levels = ofi_levels
        self.book = BookState()
        self.trades: deque[Trade] = deque()
        self.mid_history: deque[tuple[float, float]] = deque()
        self.trade_buffer_s = trade_buffer_s
        self._prev_book: BookState | None = None
        self._ofi_accum: dict[int, float] = dict.fromkeys(ofi_levels, 0.0)
        self._connected = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ stream
    @property
    def url(self) -> str:
        streams = f"{self.symbol}@depth{self.depth_levels}@{self.depth_update_ms}ms/{self.symbol}@aggTrade"
        return f"{self.ws_base}?streams={streams}"

    def start(self) -> asyncio.Task[None]:
        self._task = asyncio.create_task(self.run(), name="binance-stream")
        return self._task

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def wait_ready(self, timeout: float = 30.0) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout=timeout)

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, close_timeout=5) as ws:
                    log.info("binance stream connected: %s", self.url)
                    backoff = 1.0
                    async for raw in ws:
                        self._handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport error
                log.warning("binance stream error (%s); reconnecting in %.1fs", exc, backoff)
                self._connected.clear()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle(self, msg: dict[str, Any]) -> None:
        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        if "depth" in stream:
            self._on_depth(data)
        elif "aggTrade" in stream:
            self._on_trade(data)

    def _on_depth(self, data: dict[str, Any]) -> None:
        now = time.time()
        new = BookState(
            ts=now,
            bids=[BookLevel(float(p), float(q)) for p, q in data.get("bids", [])],
            asks=[BookLevel(float(p), float(q)) for p, q in data.get("asks", [])],
        )
        if not new.bids or not new.asks:
            return
        if self._prev_book is not None:
            for levels in self.ofi_levels:
                self._ofi_accum[levels] += order_flow_imbalance(self._prev_book, new, levels)
        self._prev_book = new
        self.book = new
        self.mid_history.append((now, new.mid))
        self._trim(now)
        self._connected.set()

    def _on_trade(self, data: dict[str, Any]) -> None:
        ts = float(data.get("T", time.time() * 1000)) / 1000.0
        self.trades.append(
            Trade(ts=ts, price=float(data["p"]), qty=float(data["q"]), buyer_maker=bool(data["m"]))
        )
        self._trim(time.time())

    def _trim(self, now: float) -> None:
        cutoff = now - self.trade_buffer_s
        while self.trades and self.trades[0].ts < cutoff:
            self.trades.popleft()
        while self.mid_history and self.mid_history[0][0] < cutoff:
            self.mid_history.popleft()

    # --------------------------------------------------------------- snapshots
    def take_snapshot(self) -> dict[str, Any] | None:
        """Snapshot the book/tape and reset the OFI accumulators."""
        book = self.book
        if not book.bids or not book.asks:
            return None
        now = time.time()
        ofi = {str(k): v for k, v in self._ofi_accum.items()}
        self._ofi_accum = dict.fromkeys(self.ofi_levels, 0.0)
        window_trades = [t for t in self.trades if t.ts >= now - 1.0]
        buy_vol = sum(t.qty for t in window_trades if not t.buyer_maker)
        sell_vol = sum(t.qty for t in window_trades if t.buyer_maker)
        return {
            "ts": now,
            "mid": book.mid,
            "microprice": book.microprice,
            "spread": book.best_ask - book.best_bid,
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "best_bid_qty": book.bids[0].qty,
            "best_ask_qty": book.asks[0].qty,
            "bid_depth": book.depth(self.depth_levels, "bid"),
            "ask_depth": book.depth(self.depth_levels, "ask"),
            "ofi": ofi,
            "book_slope_bid": book.slope(10, "bid"),
            "book_slope_ask": book.slope(10, "ask"),
            "trade_count": len(window_trades),
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "last_trade_price": self.trades[-1].price if self.trades else None,
        }


def order_flow_imbalance(prev: BookState, curr: BookState, levels: int) -> float:
    """Cont-Kukanov-Stoikov order flow imbalance summed over ``levels`` levels.

    For each level i: e_i = dBidQty(if price up/equal) - dAskQty(if price down/equal),
    with a full replacement when the level price moves away.
    """
    total = 0.0
    for i in range(levels):
        pb = prev.bids[i] if i < len(prev.bids) else None
        cb = curr.bids[i] if i < len(curr.bids) else None
        if cb is not None and pb is not None:
            if cb.price > pb.price:
                total += cb.qty
            elif cb.price == pb.price:
                total += cb.qty - pb.qty
            else:
                total -= pb.qty
        elif cb is not None:
            total += cb.qty
        elif pb is not None:
            total -= pb.qty

        pa = prev.asks[i] if i < len(prev.asks) else None
        ca = curr.asks[i] if i < len(curr.asks) else None
        if ca is not None and pa is not None:
            if ca.price < pa.price:
                total -= ca.qty
            elif ca.price == pa.price:
                total -= ca.qty - pa.qty
            else:
                total += pa.qty
        elif ca is not None:
            total -= ca.qty
        elif pa is not None:
            total += pa.qty
    return total
