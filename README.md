# pm5 — Polymarket 5-minute BTC Up/Down analysis & trading system

Streams live Binance BTC microstructure and Polymarket CLOB prices for the rolling
5-minute "Bitcoin Up or Down" markets, builds microstructure features, predicts a
calibrated probability that BTC closes the window above its open, compares it with the
market's implied probability, and paper-trades the edge with fractional-Kelly sizing.
Live execution exists but stays locked until the paper record clears an explicit gate.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,dashboard]"
cp config.example.yaml config.yaml

pm5 record --duration 3600                 # stream + store Binance/Polymarket data
pm5 backfill --days 7 --interval 1s        # or bootstrap from historical klines
pm5 train --source parquet --parquet data/parquet/klines_1s.parquet
pm5 paper                                  # log decisions, no orders
pm5 report                                 # accuracy, Brier, calibration, P&L, live gate
pm5 dashboard                              # Streamlit review UI
```

## Pipeline

### 1. Ingestion (`pm5/ingest/`)
* **Binance** (`binance.py`): `btcusdt@depth20@100ms` + `btcusdt@aggTrade` over websocket
  (`data-stream.binance.vision`, the public market-data mirror). Maintains the L2 book,
  a trade tape, and accumulates level-wise OFI between snapshots; auto-reconnects with
  exponential backoff.
* **Polymarket** (`polymarket.py`): the rolling markets use deterministic slugs
  (`btc-updown-5m-<unix window start>`), so the current market is derived from the clock
  and confirmed via the Gamma API. `MarketTracker` polls every 5s, re-subscribes the CLOB
  `market` websocket channel to the new Up/Down token ids on each roll, and persists the
  market metadata. Books are maintained incrementally from `book` + `price_change` events.

Everything lands in SQLite (`data/pm5.db`, WAL) with a Parquet export (`pm5 export`).

### 2. Features (`features.py`)
One vectorised implementation serves both training and inference — the live engine feeds
its rolling 1s snapshot buffer through the same function and takes the last row.

| Group | Features |
| --- | --- |
| Order flow imbalance | `ofi_l{1,5,10,20}_{5,60,300}s` (Cont–Kukanov–Stoikov, depth-normalised) |
| Microprice | `micro_dev_bps` (microprice − mid), `top_imbalance` |
| Momentum / volatility | `momentum_{60,300,900}s_bps`, `rv_{60,300,900}s_bps`, `ret_1s_bps` |
| Book shape | `spread_bps`, `depth_imbalance`, `log_depth`, `slope_imbalance`, `log_slope_total` |
| Trade flow | `trade_imbalance_{5,60,300}s`, `log_volume_{5,60,300}s` |
| Calendar | `tod_sin/cos`, `dow_sin/cos` (cyclical) |
| Contract state | `seconds_into_window`, `seconds_left`, `log_ret_since_open_bps`, `open_dist_z` |

`open_dist_z` is the distance from the window open scaled by the volatility of the
remaining horizon — the z-score that actually decides Up/Down.

### 3. Model (`model.py`)
LightGBM binary classifier + **post-hoc calibration** (isotonic by default, Platt/sigmoid
optional) fitted on a chronologically held-out tail of each training block.
Validation is **walk-forward** (expanding window, embargo gap between train and test so
autocorrelated rows cannot leak), scored with **Brier score and Brier skill score vs. the
base rate**, log loss and AUC; accuracy is reported but not optimised. A calibration table
(reliability curve) is stored next to the model in `model.metrics.json`.

Two targets are supported: `window` (default — matches the tradable contract: price at
window close vs. window open) and `horizon` (generic "higher in 5 minutes").

### 4. Decision & sizing (`decision.py`)
```
p_market = mid of the Up token
edge_up   = p_model       - ask_up   - cost
edge_down = (1 - p_model) - (1 - bid_up) - cost
trade only if max(edge_up, edge_down) >= min_edge
size = min(kelly_fraction * f_kelly(p, price) * bankroll, max_stake, book_size * price)
```
Guards: no trades in the first `no_trade_head_seconds` or last `no_trade_tail_seconds` of a
window, wide-spread filter, thin-book filter, one position per market.

### 5. Paper trading (`engine.py`)
Every decision — including `no_trade` ones — is written **before** the outcome is known,
with the full feature vector, `p_model`, `p_market`, edge, action, size and entry price.
A settlement loop fetches each resolved market's outcome from Gamma and writes back the
outcome and realised P&L. `pm5 report` / the dashboard then show running accuracy, Brier
score (also vs. the market's own Brier), calibration, ROI, t-stat and drawdown.

### 6. Live mode (`execution.py`)
`pm5 live --i-understand-live-risk` refuses to start unless `metrics.live_gate` passes:
at least `live_min_trades` (default 300) resolved paper trades, positive P&L with a t-stat
≥ `live_min_t_stat` (default 2.0), and a Brier score better than the market's. Orders are
marketable limit buys capped at `live_max_stake_usdc` (default $1), placed through
`py-clob-client`, and logged through the exact same pipeline as paper mode.

Credentials come from the environment only: `POLYMARKET_PRIVATE_KEY`,
`POLYMARKET_FUNDER_ADDRESS`, optionally `POLYMARKET_API_KEY/SECRET/PASSPHRASE`.

## Dashboard
`pm5 dashboard` (Streamlit): summary metrics, cumulative P&L, reliability curve,
p_model vs p_market scatter coloured by outcome, and the raw decision log.

## Caveats
* Polymarket resolves these markets from the **Chainlink BTC/USD stream**, not Binance
  spot. Binance is a proxy; expect a small basis, and label from Gamma resolutions (which
  the settlement loop does) rather than from Binance prices.
* Klines backfill only supports price-derived features; microstructure features require
  recorded stream data, so train on recorded data before trusting live numbers.
* `data-api.binance.vision` is used because `api.binance.com` is geo-blocked in some
  regions; point `binance.rest_base`/`ws_base` elsewhere if you prefer.
* A positive walk-forward Brier skill score is necessary but not sufficient — the edge must
  survive the spread, and the live gate exists precisely because it usually does not.
