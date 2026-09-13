"""Calibrated probability model with walk-forward validation.

Evaluation is proper-scoring-rule based (Brier score / log loss vs. the
climatological base rate) rather than accuracy, and every fold is trained on
strictly past data with an embargo gap to respect autocorrelation.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pm5.features import feature_columns

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    n_train: int
    n_test: int
    base_rate: float
    brier: float
    brier_baseline: float
    brier_skill_score: float
    log_loss: float
    auc: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    features: list[str]
    target: str
    trained_at: float
    metrics: dict[str, Any] = field(default_factory=dict)
    version: str = "v1"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X[self.features])[:, 1]
        return apply_calibrator(self.calibrator, raw)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        path.with_suffix(".metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        with Path(path).open("rb") as fh:
            bundle = pickle.load(fh)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle


def apply_calibrator(calibrator: Any, raw: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(raw), 1e-4, 1 - 1e-4)
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    if isinstance(calibrator, CalibratedClassifierCV):
        return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    raise TypeError(f"unsupported calibrator {type(calibrator)}")


def _make_estimator(params: dict[str, Any]) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(**params)


def fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray) -> Any:
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw, y)
        return cal
    if method in ("sigmoid", "platt"):
        cal = LogisticRegression(C=1e6, solver="lbfgs")
        cal.fit(raw.reshape(-1, 1), y)
        return cal
    raise ValueError(f"unknown calibration method: {method}")


def walk_forward_splits(
    ts: np.ndarray, n_folds: int = 6, embargo_seconds: float = 900.0, min_train_frac: float = 0.3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits ordered in time, with an embargo gap between them."""
    order = np.argsort(ts)
    ts_sorted = ts[order]
    n = len(ts_sorted)
    if n < n_folds * 10:
        raise ValueError(f"not enough rows ({n}) for {n_folds} folds")
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    splits = []
    for i in range(n_folds):
        train_end_idx = bounds[i]
        test_start_idx = bounds[i]
        test_end_idx = bounds[i + 1]
        train_cutoff = ts_sorted[train_end_idx - 1] - embargo_seconds
        train_idx = order[: train_end_idx][ts_sorted[:train_end_idx] <= train_cutoff]
        test_idx = order[test_start_idx:test_end_idx]
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _split_train_calib(
    idx: np.ndarray, ts: np.ndarray, calib_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    ordered = idx[np.argsort(ts[idx])]
    cut = int(len(ordered) * (1 - calib_fraction))
    return ordered[:cut], ordered[cut:]


def evaluate_predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    base = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_base = float(np.mean((y - base) ** 2))
    metrics = {
        "base_rate": base,
        "brier": brier,
        "brier_baseline": brier_base,
        "brier_skill_score": float(1 - brier / brier_base) if brier_base > 0 else 0.0,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
    }
    metrics["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return metrics


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_train(
    frame: pd.DataFrame,
    target: str = "window",
    n_folds: int = 6,
    calibration_method: str = "isotonic",
    calibration_fraction: float = 0.2,
    embargo_seconds: float = 900.0,
    params: dict[str, Any] | None = None,
) -> tuple[ModelBundle, list[FoldResult], pd.DataFrame]:
    """Run walk-forward evaluation, then fit the final model on all data.

    Returns the deployable bundle, per-fold metrics, and the pooled
    out-of-sample predictions (for calibration plots / P&L simulation).
    """
    params = params or {}
    features = feature_columns(frame)
    X = frame[features]
    y = frame["label"].to_numpy(dtype=int)
    ts = frame["ts"].to_numpy(dtype=float)

    fold_results: list[FoldResult] = []
    oos_rows: list[pd.DataFrame] = []

    for i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(ts, n_folds=n_folds, embargo_seconds=embargo_seconds)
    ):
        fit_idx, calib_idx = _split_train_calib(train_idx, ts, calibration_fraction)
        if len(calib_idx) < 30 or len(np.unique(y[calib_idx])) < 2:
            fit_idx, calib_idx = train_idx, train_idx
        est = _make_estimator(params)
        est.fit(X.iloc[fit_idx], y[fit_idx])
        raw_calib = est.predict_proba(X.iloc[calib_idx])[:, 1]
        calibrator = fit_calibrator(calibration_method, raw_calib, y[calib_idx])
        raw_test = est.predict_proba(X.iloc[test_idx])[:, 1]
        p_test = apply_calibrator(calibrator, raw_test)

        m = evaluate_predictions(y[test_idx], p_test)
        fold_results.append(
            FoldResult(
                fold=i,
                train_start=float(ts[train_idx].min()),
                train_end=float(ts[train_idx].max()),
                test_start=float(ts[test_idx].min()),
                test_end=float(ts[test_idx].max()),
                n_train=len(train_idx),
                n_test=len(test_idx),
                base_rate=m["base_rate"],
                brier=m["brier"],
                brier_baseline=m["brier_baseline"],
                brier_skill_score=m["brier_skill_score"],
                log_loss=m["log_loss"],
                auc=m["auc"],
                accuracy=m["accuracy"],
            )
        )
        oos_rows.append(
            pd.DataFrame(
                {
                    "ts": ts[test_idx],
                    "fold": i,
                    "p_model": p_test,
                    "p_raw": raw_test,
                    "label": y[test_idx],
                }
            )
        )
        log.info(
            "fold %d: n_train=%d n_test=%d brier=%.4f bss=%+.4f auc=%.3f",
            i, len(train_idx), len(test_idx), m["brier"], m["brier_skill_score"], m["auc"],
        )

    oos = pd.concat(oos_rows).sort_values("ts").reset_index(drop=True) if oos_rows else pd.DataFrame()

    # Final model: fit on everything, calibrate on the most recent chronological slice.
    all_idx = np.argsort(ts)
    fit_idx, calib_idx = _split_train_calib(all_idx, ts, calibration_fraction)
    final = _make_estimator(params)
    final.fit(X.iloc[fit_idx], y[fit_idx])
    calibrator = fit_calibrator(
        calibration_method, final.predict_proba(X.iloc[calib_idx])[:, 1], y[calib_idx]
    )

    pooled = evaluate_predictions(oos["label"].to_numpy(), oos["p_model"].to_numpy()) if len(oos) else {}
    metrics = {
        "pooled_oos": pooled,
        "folds": [f.to_dict() for f in fold_results],
        "n_rows": int(len(frame)),
        "features": features,
        "calibration_method": calibration_method,
        "calibration_table": (
            calibration_table(oos["label"].to_numpy(), oos["p_model"].to_numpy()).to_dict("records")
            if len(oos)
            else []
        ),
    }
    bundle = ModelBundle(
        model=final,
        calibrator=calibrator,
        features=features,
        target=target,
        trained_at=float(pd.Timestamp.utcnow().timestamp()),
        metrics=metrics,
    )
    return bundle, fold_results, oos
