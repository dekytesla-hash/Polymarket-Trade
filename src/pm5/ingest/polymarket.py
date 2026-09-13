"""Polymarket ingestion: rolling 5-minute BTC market discovery + CLOB book stream.

The 5-minute BTC Up/Down markets use deterministic slugs of the form
``btc-updown-5m-<unix window start>``, so the current market can be resolved
directly from the clock and confirmed through the Gamma API. New markets roll
every 5 minutes and the CLOB websocket subscription is re-established each time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import websockets

log = logging.getLogger(__name__)

USER_AGENT = "pm5/0.1 (+https://github.com/)"


@dataclass
class Market:
    slug: str
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    window_start: float
    window_end: float
    tick_size: float = 0.01
    min_size: float = 5.0

    def token_for(self, side: str) -> str:
        return self.up_token_id if side.lower() == "up" else self.down_token_id

    @property
    def seconds_left(self) -> float:
        return self.window_end - time.time()

    def as_row(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "condition_id": self.condition_id,
            "question": self.question,
            "up_token_id": self.up_token_id,
            "down_token_id": self.down_token_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


@dataclass
class BookTop:
    token_id: str
    best_bid: float | None
    best_ask: float | None
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float | None:
        if self.best_bid is None and self.best_ask is None:
            return None
        if self.best_bid is None:
            return self.best_ask
        if self.best_ask is None:
            return self.best_bid
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return 1.0
        return self.best_ask - self.best_bid


def window_start_for(ts: float, window_seconds: int = 300) -> int:
    return int(ts // window_seconds) * window_seconds


class GammaClient:
    """Discovers the current/next rolling market via the Gamma API."""

    def __init__(
        self,
        base: str = "https://gamma-api.polymarket.com",
        slug_prefix: str = "btc-updown-5m",
        window_seconds: int = 300,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self.slug_prefix = slug_prefix
        self.window_seconds = window_seconds
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> GammaClient:
        await self._ensure_session()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    def slug_for(self, ts: float) -> str:
        return f"{self.slug_prefix}-{window_start_for(ts, self.window_seconds)}"

    async def fetch_market(self, slug: str) -> Market | None:
        session = await self._ensure_session()
        url = f"{self.base}/events"
        async with session.get(url, params={"slug": slug}, timeout=aiohttp.ClientTimeout(total=15)) as r:
            r.raise_for_status()
            events = await r.json()
        if not events or not events[0].get("markets"):
            return None
        return self._parse(slug, events[0]["markets"][0])

    def _parse(self, slug: str, m: dict[str, Any]) -> Market | None:
        token_ids = m.get("clobTokenIds")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not token_ids or len(token_ids) < 2:
            return None
        up_idx = 0
        if outcomes:
            up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), 0)
        down_idx = 1 - up_idx
        start = float(slug.rsplit("-", 1)[-1])
        return Market(
            slug=slug,
            condition_id=m.get("conditionId", ""),
            question=m.get("question", ""),
            up_token_id=token_ids[up_idx],
            down_token_id=token_ids[down_idx],
            window_start=start,
            window_end=start + self.window_seconds,
            tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
            min_size=float(m.get("orderMinSize") or 5.0),
        )

    async def current_market(self, ts: float | None = None) -> Market | None:
        ts = ts if ts is not None else time.time()
        return await self.fetch_market(self.slug_for(ts))

    async def resolution(self, slug: str) -> int | None:
        """Return 1 if the market resolved Up, 0 if Down, None if unresolved.

        Falls back to the settled outcome prices, which snap to 0/1 on resolution.
        """
        session = await self._ensure_session()
        async with session.get(
            f"{self.base}/events", params={"slug": slug}, timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            r.raise_for_status()
            events = await r.json()
        if not events or not events[0].get("markets"):
            return None
        m = events[0]["markets"][0]
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if not prices:
            return None
        if not (m.get("closed") or m.get("umaResolutionStatus") == "resolved"):
            return None
        up_idx = next((i for i, o in enumerate(outcomes or []) if str(o).lower() == "up"), 0)
        up_price = float(prices[up_idx])
        if up_price >= 0.99:
            return 1
        if up_price <= 0.01:
            return 0
        return None


class ClobClient:
    """REST helpers for the CLOB order book."""

    def __init__(
        self, base: str = "https://clob.polymarket.com", session: aiohttp.ClientSession | None = None
    ) -> None:
        self.base = base.rstrip("/")
        self._session = session
        self._owns_session = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def book(self, token_id: str) -> BookTop:
        session = await self._ensure_session()
        async with session.get(
            f"{self.base}/book", params={"token_id": token_id}, timeout=aiohttp.ClientTimeout(total=15)
        ) as r:
            r.raise_for_status()
            data = await r.json()
        return parse_book(token_id, data.get("bids", []), data.get("asks", []))


def parse_book(token_id: str, bids: list[dict[str, Any]], asks: list[dict[str, Any]]) -> BookTop:
    """CLOB books are returned worst-to-best; the top of book is the last entry."""
    best_bid = max((float(b["price"]) for b in bids), default=None)
    best_ask = min((float(a["price"]) for a in asks), default=None)
    bid_size = sum(float(b["size"]) for b in bids if float(b["price"]) == best_bid) if bids else 0.0
    ask_size = sum(float(a["size"]) for a in asks if float(a["price"]) == best_ask) if asks else 0.0
    return BookTop(token_id, best_bid, best_ask, bid_size, ask_size)


class ClobMarketStream:
    """Websocket subscription to the CLOB ``market`` channel for a set of tokens.

    Call :meth:`resubscribe` when the rolling market changes; the socket is
    reconnected with the new asset ids.
    """

    def __init__(self, url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market") -> None:
        self.url = url
        self.books: dict[str, BookTop] = {}
        self.last_update: float = 0.0
        self._token_ids: list[str] = []
        self._levels: dict[str, dict[str, dict[float, float]]] = {}
        self._task: asyncio.Task[None] | None = None
        self._restart = asyncio.Event()

    def start(self) -> asyncio.Task[None]:
        self._task = asyncio.create_task(self.run(), name="clob-stream")
        return self._task

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def resubscribe(self, token_ids: list[str]) -> None:
        if token_ids == self._token_ids:
            return
        self._token_ids = list(token_ids)
        self.books = {}
        self._levels = {}
        self._restart.set()

    async def run(self) -> None:
        backoff = 1.0
        while True:
            if not self._token_ids:
                await asyncio.sleep(0.5)
                continue
            tokens = list(self._token_ids)
            self._restart.clear()
            try:
                async with websockets.connect(self.url, ping_interval=10, close_timeout=5) as ws:
                    await ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))
                    log.info("clob stream subscribed to %d tokens", len(tokens))
                    backoff = 1.0
                    await self._consume(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport error
                log.warning("clob stream error (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _consume(self, ws: Any) -> None:
        restart_task = asyncio.create_task(self._restart.wait())
        try:
            while True:
                recv_task = asyncio.create_task(ws.recv())
                done, _ = await asyncio.wait(
                    {recv_task, restart_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if restart_task in done:
                    recv_task.cancel()
                    return
                raw = recv_task.result()
                if raw in ("PONG", "PING"):
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for msg in payload if isinstance(payload, list) else [payload]:
                    self._handle(msg)
        finally:
            restart_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await restart_task

    def _handle(self, msg: dict[str, Any]) -> None:
        event = msg.get("event_type")
        token_id = msg.get("asset_id") or msg.get("market")
        if event == "book" and token_id:
            self._levels[token_id] = {
                "bids": {float(b["price"]): float(b["size"]) for b in msg.get("bids", [])},
                "asks": {float(a["price"]): float(a["size"]) for a in msg.get("asks", [])},
            }
            self._rebuild(token_id)
        elif event == "price_change":
            for change in msg.get("changes", []) or []:
                tid = change.get("asset_id") or token_id
                if not tid:
                    continue
                side = "bids" if str(change.get("side", "")).upper() == "BUY" else "asks"
                levels = self._levels.setdefault(tid, {"bids": {}, "asks": {}})
                price, size = float(change["price"]), float(change["size"])
                if size <= 0:
                    levels[side].pop(price, None)
                else:
                    levels[side][price] = size
                self._rebuild(tid)

    def _rebuild(self, token_id: str) -> None:
        levels = self._levels.get(token_id)
        if levels is None:
            return
        bids = [{"price": p, "size": s} for p, s in levels["bids"].items() if s > 0]
        asks = [{"price": p, "size": s} for p, s in levels["asks"].items() if s > 0]
        self.books[token_id] = parse_book(token_id, bids, asks)
        self.last_update = time.time()


class MarketTracker:
    """Keeps the current rolling market and its CLOB subscription up to date."""

    def __init__(self, gamma: GammaClient, stream: ClobMarketStream, poll_interval: float = 5.0) -> None:
        self.gamma = gamma
        self.stream = stream
        self.poll_interval = poll_interval
        self.market: Market | None = None
        self._task: asyncio.Task[None] | None = None
        self._listeners: list[Any] = []

    def on_new_market(self, callback: Any) -> None:
        self._listeners.append(callback)

    def start(self) -> asyncio.Task[None]:
        self._task = asyncio.create_task(self.run(), name="market-tracker")
        return self._task

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def refresh(self) -> Market | None:
        expected = self.gamma.slug_for(time.time())
        if self.market is not None and self.market.slug == expected:
            return self.market
        try:
            market = await self.gamma.current_market()
        except Exception as exc:  # noqa: BLE001 - transient API errors must not kill the loop
            log.warning("gamma market lookup failed: %s", exc)
            return self.market
        if market is None:
            log.warning("no market found for slug %s", expected)
            return self.market
        self.market = market
        self.stream.resubscribe([market.up_token_id, market.down_token_id])
        for cb in self._listeners:
            result = cb(market)
            if asyncio.iscoroutine(result):
                await result
        log.info("tracking market %s (%s)", market.slug, market.question)
        return market

    async def run(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.poll_interval)
