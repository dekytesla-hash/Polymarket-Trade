"""Configuration for the pm5 system.

Values are loaded from a YAML file (see config.example.yaml) with environment
variable overrides for anything secret.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class BinanceConfig:
    symbol: str = "btcusdt"
    # Public market-data mirror; api.binance.com is geo-blocked in some regions.
    rest_base: str = "https://data-api.binance.vision"
    ws_base: str = "wss://data-stream.binance.vision/stream"
    depth_levels: int = 20
    depth_update_ms: int = 100


@dataclass
class PolymarketConfig:
    gamma_base: str = "https://gamma-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    clob_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    slug_prefix: str = "btc-updown-5m"
    window_seconds: int = 300
    # Seconds before window end after which we no longer open new positions.
    no_trade_tail_seconds: int = 20
    # Seconds after window start before we are willing to trade.
    no_trade_head_seconds: int = 15


@dataclass
class FeatureConfig:
    ofi_levels: tuple[int, ...] = (1, 5, 10, 20)
    momentum_windows_s: tuple[int, ...] = (60, 300, 900)
    vol_windows_s: tuple[int, ...] = (60, 300, 900)
    book_slope_levels: int = 10


@dataclass
class ModelConfig:
    horizon_seconds: int = 300
    # Number of walk-forward folds used for out-of-sample evaluation.
    n_folds: int = 6
    # Fraction of each training block held out (chronologically) for calibration.
    calibration_fraction: float = 0.2
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid"
    embargo_seconds: int = 900
    model_dir: Path = Path("models")
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "n_estimators": 400,
            "verbose": -1,
        }
    )


@dataclass
class TradingConfig:
    # Minimum |p_model - p_market| required before considering a trade.
    min_edge: float = 0.04
    # Round-trip cost assumption (taker spread + slippage), in probability units.
    cost: float = 0.01
    kelly_fraction: float = 0.25
    max_stake_usdc: float = 5.0
    bankroll_usdc: float = 500.0
    # Refuse to trade markets whose book is wider than this.
    max_spread: float = 0.06
    min_book_size: float = 20.0
    # Live trading is only allowed when the paper-trading record clears these gates.
    live_min_trades: int = 300
    live_min_t_stat: float = 2.0
    live_max_stake_usdc: float = 1.0


@dataclass
class StorageConfig:
    db_path: Path = Path("data/pm5.db")
    parquet_dir: Path = Path("data/parquet")
    snapshot_interval_s: float = 1.0


@dataclass
class Config:
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mode: str = "paper"  # "paper" | "live"

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
        cfg = _from_dict(cls, raw)
        if os.getenv("PM5_MODE"):
            cfg.mode = os.environ["PM5_MODE"]
        return cfg


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) or (isinstance(value, dict) and is_dataclass(_resolve(f.type))):
            kwargs[f.name] = _from_dict(_resolve(f.type), value)
        elif f.type in (Path, "Path"):
            kwargs[f.name] = Path(value)
        elif isinstance(value, list) and "tuple" in str(f.type):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _resolve(tp: Any) -> Any:
    if isinstance(tp, str):
        return globals().get(tp, tp)
    return tp


@dataclass
class LiveCredentials:
    """Credentials for live CLOB execution, sourced from the environment only."""

    private_key: str
    funder_address: str | None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    chain_id: int = 137

    @classmethod
    def from_env(cls) -> LiveCredentials | None:
        pk = os.getenv("POLYMARKET_PRIVATE_KEY")
        if not pk:
            return None
        return cls(
            private_key=pk,
            funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            api_key=os.getenv("POLYMARKET_API_KEY"),
            api_secret=os.getenv("POLYMARKET_API_SECRET"),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
        )
