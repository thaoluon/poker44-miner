"""Train the Poker44 bot-detection model on the cached benchmark data.

Usage:
    python scripts/miner/train_model.py [--data-dir data/benchmark] \
        [--out poker44/model/artifacts]

Pipeline:
  1. Load every cached API chunk; each chunk group (30-40 hands) is one
     training example labeled by ``groundTruth``.
  2. Extract chunk-level features (poker44.model.features).
  3. Leave-date-out cross-validation with LightGBM, scored with the real
     subnet ``reward()`` plus AP / recall@5%FPR.
  4. Fit a final model on all data, calibrate the score mapping so the
     validator's hard 0.5 threshold lands at a low-FPR operating point.
  5. Save artifacts: LightGBM model, feature names, calibration params.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker44.model.features import chunk_features  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402


def to_miner_view(group: list[dict]) -> list[dict]:
    """Project raw benchmark hands through the validator's miner-visible
    canonicalizer so training data matches what the live miner receives.

    The transform is deterministic (sha256-seeded bucketing + action windows),
    so this exactly reproduces the validator's served payload. Any hand that
    fails to project is passed through unchanged rather than dropped.
    """
    viewed = []
    for hand in group:
        try:
            viewed.append(prepare_hand_for_miner(hand))
        except Exception:  # noqa: BLE001
            viewed.append(hand)
    return viewed

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.015,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 25,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 2000,
    "verbosity": -1,
    "seed": 44,
}


def load_dataset(data_dir: Path):
    rows, labels, dates, splits = [], [], [], []
    for date_dir in sorted(data_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for path in sorted(date_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            payload = json.loads(path.read_text())
            groups = payload.get("chunks") or []
            truth = payload.get("groundTruth") or []
            if len(groups) != len(truth):
                print(f"skip {path.name}: {len(groups)} groups vs {len(truth)} labels")
                continue
            for group, label in zip(groups, truth):
                rows.append(chunk_features(to_miner_view(group)))
                labels.append(int(label))
                dates.append(date_dir.name)
                splits.append(payload.get("split") or "train")
    feature_names = sorted({k for row in rows for k in row})
    X = np.asarray(
        [[row.get(k, 0.0) for k in feature_names] for row in rows], dtype=float
    )
    return X, np.asarray(labels), np.asarray(dates), np.asarray(splits), feature_names


def fit_model(X, y):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    return model


def calibrate(oof_scores: np.ndarray, labels: np.ndarray) -> dict:
    """Choose a raw-score pivot mapped to 0.5 (monotone piecewise-linear).

    Ranking metrics (AP, recall@FPR) are unchanged; only the hard 0.5
    threshold moves.  The pivot MUST come from out-of-fold scores — training
    scores are overconfident and put the pivot far too low.  Pivot = the
    out-of-fold human 96th percentile (~4% FPR), safely inside the
    validator's 10% hard-FPR budget with margin for distribution shift.
    """
    humans = np.sort(oof_scores[labels == 0])
    if humans.size == 0:
        return {"pivot": 0.5}
    pivot = float(np.quantile(humans, 0.98))
    pivot = min(max(pivot, 0.05), 0.95)
    return {"pivot": pivot}


def apply_calibration(scores: np.ndarray, calib: dict) -> np.ndarray:
    pivot = float(calib.get("pivot", 0.5))
    out = np.where(
        scores <= pivot,
        0.5 * scores / max(pivot, 1e-9),
        0.5 + 0.5 * (scores - pivot) / max(1.0 - pivot, 1e-9),
    )
    return np.clip(out, 0.0, 1.0)


def evaluate(scores, labels, tag):
    rew, detail = reward(scores, labels)
    ap = average_precision_score(labels, scores) if labels.any() else 0.0
    auc = roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else 0.0
    print(
        f"  {tag:24s} reward={rew:.4f} AP={ap:.4f} AUC={auc:.4f} "
        f"recall@5%FPR={detail['bot_recall']:.4f} "
        f"hard_fpr={detail['hard_fpr']:.4f} hard_recall={detail['hard_bot_recall']:.4f}"
    )
    return rew


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/benchmark")
    parser.add_argument("--out", default="poker44/model/artifacts")
    args = parser.parse_args()

    X, y, dates, splits, feature_names = load_dataset(Path(args.data_dir))
    print(
        f"dataset: {len(y)} chunk groups, {X.shape[1]} features, "
        f"{int(y.sum())} bot / {int((y == 0).sum())} human, "
        f"{len(set(dates.tolist()))} release dates"
    )

    # --- Leave-date-out CV (big dates individually, small dates in one fold).
    unique_dates = sorted(set(dates.tolist()))
    big_dates = [d for d in unique_dates if (dates == d).sum() >= 50]
    small_dates = [d for d in unique_dates if d not in big_dates]
    folds = [[d] for d in big_dates] + ([small_dates] if small_dates else [])

    # Pass 1: collect raw out-of-fold scores.
    oof_raw = np.full(len(y), np.nan)
    fold_masks = []
    for fold_dates in folds:
        mask = np.isin(dates, fold_dates)
        fold_masks.append((fold_dates, mask))
        model = fit_model(X[~mask], y[~mask])
        oof_raw[mask] = model.predict_proba(X[mask])[:, 1]
    assert not np.isnan(oof_raw).any()

    # Pass 2: single global pivot from OOF human scores, then evaluate.
    calib = calibrate(oof_raw, y)
    oof_scores = apply_calibration(oof_raw, calib)
    print(f"\ncalibration pivot (from OOF): {calib['pivot']:.4f}")
    print("\nleave-date-out cross-validation:")
    for fold_dates, mask in fold_masks:
        tag = fold_dates[0] if len(fold_dates) == 1 else f"small×{len(fold_dates)}"
        evaluate(oof_scores[mask], y[mask], tag)

    print("\npooled out-of-fold:")
    evaluate(oof_scores, y, "ALL (oof)")

    # --- Also respect the API's own train/validation split as a sanity check.
    if (splits == "validation").any():
        tr, va = splits != "validation", splits == "validation"
        model = fit_model(X[tr], y[tr])
        va_scores = apply_calibration(model.predict_proba(X[va])[:, 1], calib)
        print("\napi split validation:")
        evaluate(va_scores, y[va], "api validation")

    # --- Final model on everything; keep the OOF-derived pivot.
    final_model = fit_model(X, y)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(out_dir / "model.lgb.txt"))
    (out_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "calibration": calib,
                "n_train_examples": int(len(y)),
                "release_dates": unique_dates,
                "lgb_params": LGB_PARAMS,
            },
            indent=2,
        )
    )
    print(f"saved model + meta to {out_dir}")

    importances = sorted(
        zip(feature_names, final_model.feature_importances_),
        key=lambda kv: -kv[1],
    )[:20]
    print("\ntop features:")
    for name, imp in importances:
        print(f"  {imp:6d}  {name}")


if __name__ == "__main__":
    main()
