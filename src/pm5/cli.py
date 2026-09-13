"""Command line interface: pm5 <command>."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from pm5.config import Config, LiveCredentials
from pm5.storage import Storage


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_record(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.engine import Engine

    with Storage(cfg.storage.db_path) as storage:
        engine = Engine(cfg, storage, model=None, record_only=True)
        asyncio.run(engine.run(duration_s=args.duration))
    return 0


def cmd_backfill(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.dataset import backfill_klines, save_parquet

    df = backfill_klines(
        symbol=cfg.binance.symbol.upper(),
        days=args.days,
        interval=args.interval,
        rest_base=cfg.binance.rest_base,
    )
    path = save_parquet(df, args.out or cfg.storage.parquet_dir / f"klines_{args.interval}.parquet")
    print(f"wrote {len(df)} rows to {path}")
    return 0


def _load_frame(args: argparse.Namespace, cfg: Config):
    import pandas as pd

    from pm5.dataset import build_training_frame, load_recorded_snapshots

    if args.source == "recorded":
        with Storage(cfg.storage.db_path) as storage:
            snaps = load_recorded_snapshots(storage)
    else:
        snaps = pd.read_parquet(args.parquet or cfg.storage.parquet_dir / "klines_1s.parquet")
    if snaps.empty:
        raise SystemExit("no snapshots available; run 'pm5 record' or 'pm5 backfill' first")
    return build_training_frame(
        snaps,
        target=args.target,
        window_seconds=cfg.polymarket.window_seconds,
        horizon_seconds=cfg.model.horizon_seconds,
    )


def cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.model import walk_forward_train

    frame = _load_frame(args, cfg)
    print(f"training on {len(frame)} rows, {frame['label'].mean():.3f} base rate")
    bundle, folds, oos = walk_forward_train(
        frame,
        target=args.target,
        n_folds=cfg.model.n_folds,
        calibration_method=cfg.model.calibration_method,
        calibration_fraction=cfg.model.calibration_fraction,
        embargo_seconds=cfg.model.embargo_seconds,
        params=cfg.model.lgbm_params,
    )
    out = Path(args.out) if args.out else cfg.model.model_dir / "model.pkl"
    bundle.save(out)
    pooled = bundle.metrics.get("pooled_oos", {})
    print(json.dumps({"model": str(out), "folds": len(folds), "pooled_oos": pooled}, indent=2))
    if args.oos_out:
        oos.to_parquet(args.oos_out, index=False)
    return 0


def cmd_trade(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.engine import Engine, assert_live_allowed, load_model

    cfg.mode = args.mode
    model = load_model(cfg, args.model)
    if model is None:
        raise SystemExit("no trained model found; run 'pm5 train' first")
    executor = None
    with Storage(cfg.storage.db_path) as storage:
        if cfg.mode == "live":
            if not args.i_understand_live_risk:
                raise SystemExit("live mode requires --i-understand-live-risk")
            try:
                assert_live_allowed(cfg, storage)
            except RuntimeError as exc:
                raise SystemExit(f"live mode blocked: {exc}") from None
            creds = LiveCredentials.from_env()
            if creds is None:
                raise SystemExit("live mode requires POLYMARKET_PRIVATE_KEY in the environment")
            from pm5.execution import ClobExecutor

            executor = ClobExecutor(cfg, creds)
        engine = Engine(cfg, storage, model=model, executor=executor)
        asyncio.run(engine.run(duration_s=args.duration))
    return 0


def cmd_settle(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.engine import Engine

    with Storage(cfg.storage.db_path) as storage:
        engine = Engine(cfg, storage, model=None, record_only=True)

        async def _run() -> int:
            try:
                return await engine.settle_pending()
            finally:
                await engine.gamma.close()

        n = asyncio.run(_run())
    print(f"settled {n} decisions")
    return 0


def cmd_report(args: argparse.Namespace, cfg: Config) -> int:
    from pm5.metrics import calibration, decisions_frame, live_gate, summarize

    with Storage(cfg.storage.db_path) as storage:
        df = decisions_frame(storage, mode=args.mode)
        if df.empty:
            print("no decisions logged yet")
            return 0
        summary = summarize(df)
        print(json.dumps(summary, indent=2, default=float))
        print("\ncalibration:")
        print(calibration(df).to_string(index=False))
        ok, reason = live_gate(df, cfg.trading.live_min_trades, cfg.trading.live_min_t_stat)
        print(f"\nlive gate: {'PASS' if ok else 'BLOCKED'} - {reason}")
    return 0


def cmd_export(args: argparse.Namespace, cfg: Config) -> int:
    with Storage(cfg.storage.db_path) as storage:
        paths = storage.export_parquet(args.out or cfg.storage.parquet_dir)
    print("\n".join(str(p) for p in paths))
    return 0


def cmd_dashboard(args: argparse.Namespace, cfg: Config) -> int:
    import subprocess

    dashboard = Path(__file__).with_name("dashboard.py")
    env = {
        **os.environ,
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard),
            "--server.port",
            str(args.port),
        ],
        env=env,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pm5", description=__doc__)
    p.add_argument("-c", "--config", default=None, help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="stream and store Binance + Polymarket data")
    rec.add_argument("--duration", type=float, default=None, help="seconds to run (default: forever)")
    rec.set_defaults(func=cmd_record)

    bf = sub.add_parser("backfill", help="download historical Binance klines")
    bf.add_argument("--days", type=float, default=7.0)
    bf.add_argument("--interval", default="1s", choices=["1s", "1m", "5m"])
    bf.add_argument("--out", default=None)
    bf.set_defaults(func=cmd_backfill)

    tr = sub.add_parser("train", help="walk-forward train + calibrate the model")
    tr.add_argument("--source", default="recorded", choices=["recorded", "parquet"])
    tr.add_argument("--parquet", default=None)
    tr.add_argument("--target", default="window", choices=["window", "horizon"])
    tr.add_argument("--out", default=None)
    tr.add_argument("--oos-out", default=None, help="write out-of-sample predictions to parquet")
    tr.set_defaults(func=cmd_train)

    pa = sub.add_parser("paper", help="run the paper-trading engine")
    pa.add_argument("--duration", type=float, default=None)
    pa.add_argument("--model", default=None)
    pa.set_defaults(func=cmd_trade, mode="paper", i_understand_live_risk=False)

    lv = sub.add_parser("live", help="run live execution (gated on paper results)")
    lv.add_argument("--duration", type=float, default=None)
    lv.add_argument("--model", default=None)
    lv.add_argument("--i-understand-live-risk", action="store_true")
    lv.set_defaults(func=cmd_trade, mode="live")

    st = sub.add_parser("settle", help="fetch outcomes for resolved markets")
    st.set_defaults(func=cmd_settle)

    rp = sub.add_parser("report", help="print accuracy, Brier score, calibration and P&L")
    rp.add_argument("--mode", default=None, choices=["paper", "live"])
    rp.set_defaults(func=cmd_report)

    ex = sub.add_parser("export", help="export all tables to Parquet")
    ex.add_argument("--out", default=None)
    ex.set_defaults(func=cmd_export)

    db = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    db.add_argument("--port", type=int, default=8501)
    db.set_defaults(func=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = Config.load(args.config)
    return int(args.func(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
