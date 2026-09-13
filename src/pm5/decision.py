"""Edge computation and fractional-Kelly position sizing."""

from __future__ import annotations

from dataclasses import dataclass

from pm5.config import TradingConfig


@dataclass
class Quote:
    """Executable prices for the Up token (Down is the 1 - price mirror)."""

    best_bid: float | None
    best_ask: float | None
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return self.best_bid if self.best_ask is None else self.best_ask
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return 1.0
        return self.best_ask - self.best_bid


@dataclass
class Decision:
    action: str  # "buy_up" | "buy_down" | "no_trade"
    reason: str
    p_model: float
    p_market: float
    edge: float
    size_usdc: float = 0.0
    entry_price: float | None = None
    kelly_fraction: float = 0.0


def kelly_fraction(p: float, price: float) -> float:
    """Kelly stake fraction for a binary contract bought at ``price``.

    Payoff per $1: win 1/price - 1, lose 1. f* = (p*(1-price) - (1-p)*price) / (1-price)
    """
    price = min(max(price, 1e-6), 1 - 1e-6)
    b = (1.0 - price) / price
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def decide(
    p_model: float,
    quote: Quote,
    cfg: TradingConfig,
    seconds_left: float | None = None,
    no_trade_tail_seconds: float = 20.0,
    max_stake_override: float | None = None,
) -> Decision:
    """Compare the model probability with the market and size a trade.

    ``p_market`` is the mid of the Up token. Trading UP pays ``best_ask``;
    trading DOWN pays ``1 - best_bid`` for the Down token (equivalently selling
    Up at the bid). Costs are charged against the edge before sizing.
    """
    p_market = quote.mid
    if p_market is None:
        return Decision("no_trade", "no_market_quote", p_model, float("nan"), 0.0)
    edge = p_model - p_market

    if seconds_left is not None and seconds_left <= no_trade_tail_seconds:
        return Decision("no_trade", "too_close_to_expiry", p_model, p_market, edge)
    if quote.spread > cfg.max_spread:
        return Decision("no_trade", "spread_too_wide", p_model, p_market, edge)

    max_stake = cfg.max_stake_usdc if max_stake_override is None else max_stake_override

    # Buying UP at the ask.
    up_price = quote.best_ask
    up_edge = (p_model - up_price - cfg.cost) if up_price is not None else -1.0
    # Buying DOWN: the Down token's ask is ~ 1 - Up's bid.
    down_price = (1.0 - quote.best_bid) if quote.best_bid is not None else None
    down_edge = ((1.0 - p_model) - down_price - cfg.cost) if down_price is not None else -1.0

    if max(up_edge, down_edge) < cfg.min_edge:
        return Decision("no_trade", "edge_below_threshold", p_model, p_market, edge)

    if up_edge >= down_edge:
        action, price, p_win, avail = "buy_up", up_price, p_model, quote.ask_size
    else:
        action, price, p_win, avail = "buy_down", down_price, 1.0 - p_model, quote.bid_size

    assert price is not None
    f = kelly_fraction(p_win, price) * cfg.kelly_fraction
    size = min(f * cfg.bankroll_usdc, max_stake)
    if avail and size > avail * price:
        size = avail * price
    if size <= 0:
        return Decision(action, "zero_size", p_model, p_market, edge, 0.0, price, f)
    if avail < cfg.min_book_size:
        return Decision("no_trade", "insufficient_book_size", p_model, p_market, edge)
    return Decision(action, "trade", p_model, p_market, edge, round(size, 4), price, f)


def realized_pnl(action: str, entry_price: float, size_usdc: float, outcome: int) -> float:
    """P&L in USDC for a resolved market. ``outcome`` is 1 for Up, 0 for Down."""
    if action == "no_trade" or size_usdc <= 0 or entry_price <= 0:
        return 0.0
    shares = size_usdc / entry_price
    won = (action == "buy_up" and outcome == 1) or (action == "buy_down" and outcome == 0)
    return shares - size_usdc if won else -size_usdc
