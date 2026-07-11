"""Quick leave-date-out hyperparameter sweep + seed-ensemble check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker44.score.scoring import _recall_at_fpr  # noqa: E402
from scripts.miner.train_model import load_dataset  # noqa: E402

BASE = {
    "objective": "binary",
    "verbosity": -1,
    "subsample_freq": 1,
}

CONFIGS = {
    "current": dict(learning_rate=0.03, num_leaves=31, max_depth=6,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0, n_estimators=600),
    "slow_deep": dict(learning_rate=0.015, num_leaves=63, max_depth=-1,
                      min_child_samples=15, subsample=0.8, colsample_bytree=0.6,
                      reg_alpha=0.1, reg_lambda=1.0, n_estimators=1500),
    "slow_shallow": dict(learning_rate=0.015, num_leaves=15, max_depth=4,
                         min_child_samples=25, subsample=0.9, colsample_bytree=0.5,
                         reg_alpha=0.5, reg_lambda=2.0, n_estimators=2000),
    "fast_reg": dict(learning_rate=0.05, num_leaves=31, max_depth=5,
                     min_child_samples=30, subsample=0.7, colsample_bytree=0.5,
                     reg_alpha=1.0, reg_lambda=5.0, n_estimators=500),
}

SEEDS = [44, 45, 46, 47, 48]


def oof_scores(X, y, dates, params, seeds):
    unique_dates = sorted(set(dates.tolist()))
    big = [d for d in unique_dates if (dates == d).sum() >= 50]
    small = [d for d in unique_dates if d not in big]
    folds = [[d] for d in big] + ([small] if small else [])
    oof = np.zeros(len(y))
    for fold_dates in folds:
        mask = np.isin(dates, fold_dates)
        preds = np.zeros(mask.sum())
        for seed in seeds:
            model = lgb.LGBMClassifier(**BASE, **params, seed=seed)
            model.fit(X[~mask], y[~mask])
            preds += model.predict_proba(X[mask])[:, 1]
        oof[mask] = preds / len(seeds)
    return oof


def report(tag, scores, y):
    ap = average_precision_score(y, scores)
    recall, _ = _recall_at_fpr(scores, y, max_fpr=0.05)
    partial = 0.35 * ap + 0.30 * recall
    print(f"{tag:28s} AP={ap:.4f} recall@5%FPR={recall:.4f} "
          f"rank_reward(0.65max)={partial:.4f}")
    return partial


def main():
    X, y, dates, _, _ = load_dataset(Path("data/benchmark"))
    print(f"{len(y)} examples, {X.shape[1]} features\n")

    results = {}
    for name, params in CONFIGS.items():
        scores = oof_scores(X, y, dates, params, seeds=[44])
        results[name] = (report(name, scores, y), params)

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nbest single-seed config: {best_name}; testing 5-seed ensemble:")
    best_params = results[best_name][1]
    ens = oof_scores(X, y, dates, best_params, seeds=SEEDS)
    report(f"{best_name} x5 seeds", ens, y)

    Path("data/tune_result.json").write_text(
        json.dumps({"best": best_name, "params": best_params})
    )


if __name__ == "__main__":
    main()
