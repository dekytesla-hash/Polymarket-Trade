"""Live order execution via the Polymarket CLOB API.

Deliberately minimal: marketable limit orders, one per market, hard-capped
stake. Requires ``py-clob-client`` and credentials in the environment; the
engine only constructs an executor after the live gate passes.
"""

from __future__ import annotations

import logging
from typing import Any

from pm5.config import Config, LiveCredentials
from pm5.decision import Decision
from pm5.ingest.polymarket import Market

log = logging.getLogger(__name__)


class ClobExecutor:
    def __init__(self, cfg: Config, creds: LiveCredentials) -> None:
        try:
            from py_clob_client.client import ClobClient as _ClobClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("live mode requires: pip install '.[live]'") from exc

        self.cfg = cfg
        self.client = _ClobClient(
            host=cfg.polymarket.clob_base,
            key=creds.private_key,
            chain_id=creds.chain_id,
            signature_type=2 if creds.funder_address else 0,
            funder=creds.funder_address,
        )
        if creds.api_key and creds.api_secret and creds.api_passphrase:
            from py_clob_client.clob_types import ApiCreds

            self.client.set_api_creds(
                ApiCreds(
                    api_key=creds.api_key,
                    api_secret=creds.api_secret,
                    api_passphrase=creds.api_passphrase,
                )
            )
        else:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

    async def place(self, market: Market, dec: Decision) -> str | None:
        """Place a marketable limit buy for the chosen side. Returns the order id."""
        if dec.action == "no_trade" or dec.entry_price is None:
            return None
        stake = min(dec.size_usdc, self.cfg.trading.live_max_stake_usdc)
        token_id = market.token_for("up" if dec.action == "buy_up" else "down")
        # Cross the spread by one tick to improve the fill probability inside a
        # 5-minute window, but never pay more than the modelled fair value.
        price = min(round(dec.entry_price + market.tick_size, 2), 0.99)
        size = round(stake / price, 2)
        if size < market.min_size:
            log.info("skipping live order: size %.2f below market minimum %.2f", size, market.min_size)
            return None
        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY

            order = self.client.create_order(
                OrderArgs(price=price, size=size, side=BUY, token_id=token_id)
            )
            resp: dict[str, Any] = self.client.post_order(order)
            order_id = str(resp.get("orderID") or resp.get("orderId") or "")
            log.warning("LIVE order %s: %s %.2f shares @ %.2f", order_id, dec.action, size, price)
            return order_id or None
        except Exception as exc:  # noqa: BLE001 - never let execution kill the loop
            log.exception("live order failed: %s", exc)
            return None
