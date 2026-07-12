"""Rank-averaged multi-family ensemble (dragon-0 style) vs single LGBM.
Leave-date-out; measure reward on pooled OOF and the api-split holdout.
"""
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.score.scoring import reward
from scripts.miner.train_model import load_dataset, LGB_PARAMS, calibrate, apply_calibration

X, y, dates, splits, names = load_dataset(Path("data/benchmark"))
print(f"{len(y)} chunks, {X.shape[1]} features")

def members():
    return {
        "lgb": lgb.LGBMClassifier(**LGB_PARAMS),
        "xgb": XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                             tree_method="hist", eval_metric="logloss", random_state=44),
        "cat": CatBoostClassifier(iterations=700, depth=6, learning_rate=0.03,
                                  l2_leaf_reg=3.0, verbose=False, random_seed=44),
        "et": ExtraTreesClassifier(n_estimators=500, max_depth=14, min_samples_leaf=8,
                                   max_features=0.5, n_jobs=-1, random_state=44),
    }

ud = sorted(set(dates.tolist()))
big = [d for d in ud if (dates == d).sum() >= 50]
folds = [[d] for d in big] + [[d for d in ud if d not in big]]

# collect per-model OOF probabilities
oof = {k: np.zeros(len(y)) for k in members()}
for fd in folds:
    te = np.isin(dates, fd); tr = ~te
    for k, m in members().items():
        m.fit(X[tr], y[tr])
        oof[k][te] = m.predict_proba(X[te])[:, 1]

def rank01(s):  # within-array rank normalized to [0,1]
    return (rankdata(s) - 1) / (len(s) - 1)

def score(s, tag, mask=None):
    idx = np.ones(len(y), bool) if mask is None else mask
    cal = calibrate(s[idx], y[idx]); sc = apply_calibration(s[idx], cal)
    r, d = reward(sc, y[idx])
    print(f"  {tag:26s} reward={r:.4f} AP={average_precision_score(y[idx],sc):.4f} recall@5%FPR={d['bot_recall']:.4f}")
    return r

va = splits == "validation"
# rank-transform each model's OOF per-fold (mimics per-batch ranking at serve)
oof_rank = {k: np.zeros(len(y)) for k in members()}
for fd in folds:
    te = np.isin(dates, fd)
    for k in members():
        oof_rank[k][te] = rank01(oof[k][te])

prob_avg = np.mean([oof[k] for k in members()], axis=0)
rank_avg = np.mean([oof_rank[k] for k in members()], axis=0)

print("\nPOOLED OUT-OF-FOLD:")
score(oof["lgb"], "single LGBM (baseline)")
score(prob_avg, "4-family prob-average")
score(rank_avg, "4-family RANK-average")
print("\nAPI-SPLIT:")
score(oof["lgb"], "single LGBM", va)
score(rank_avg, "4-family RANK-average", va)
np.savez("data/ens_oof.npz", **{f"oof_{k}": oof[k] for k in members()},
         **{f"rank_{k}": oof_rank[k] for k in members()}, y=y, va=va, dates=dates)
print("\nper-model solo OOF reward:")
for k in members():
    score(oof[k], f"  {k} only")
