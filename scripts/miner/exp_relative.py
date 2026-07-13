"""Second feature view: within-batch rank-normalized features.

Ranks are invariant to monotone rescaling, so relative features survive the
confirmed benchmark->live magnitude shift (230bb->100bb). Test whether adding
this decorrelated view improves ranking on native benchmark AND on
simulated-live (rescaled to 100bb), leave-date-out.
"""
import sys, json, glob, copy, random
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.model.features import chunk_features, _safe
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from scripts.miner.train_model import LGB_PARAMS, calibrate, apply_calibration

def rescale_hand(hand, target=100.0):
    h = copy.deepcopy(hand); meta = h.get("metadata") or {}
    bb = _safe(meta.get("bb"), 0.0) or 1.0; hero = meta.get("hero_seat"); hs = 0.0
    for p in h.get("players") or []:
        if p.get("seat") == hero: hs = _safe(p.get("starting_stack")) / bb
    if hs <= 1.0: return h
    sc = target / hs
    for p in h.get("players") or []:
        if p.get("starting_stack") is not None: p["starting_stack"] = _safe(p.get("starting_stack")) * sc
    for a in h.get("actions") or []:
        for k in ("amount", "pot_before", "pot_after", "normalized_amount_bb"):
            if a.get(k) is not None: a[k] = _safe(a.get(k)) * sc
    return h

# load benchmark grouped by date
bydate = {}
for dd in sorted(Path("/root/Poker44-subnet/data/benchmark").iterdir()):
    if not dd.is_dir(): continue
    for f in sorted(dd.glob("*.json")):
        if f.name == "manifest.json": continue
        p = json.loads(f.read_text())
        if len(p.get("chunks") or []) != len(p.get("groundTruth") or []): continue
        for g, l in zip(p["chunks"], p["groundTruth"]):
            bydate.setdefault(dd.name, []).append(([prepare_hand_for_miner(h) for h in g], int(l)))

big = [d for d in bydate if len(bydate[d]) >= 50]
print(f"{len(big)} big-date batches, sizes {[len(bydate[d]) for d in big]}")

names = None
def feats_for(chunks_labels, rescale=False):
    global names
    chunks = [ (([rescale_hand(h) for h in c]) if rescale else c) for c,_ in chunks_labels ]
    F = [chunk_features(c) for c in chunks]
    if names is None: names = sorted({k for x in F for k in x})
    A = np.array([[x.get(k,0.0) for k in names] for x in F])
    # within-batch rank-normalized (relative view)
    R = np.apply_along_axis(lambda col: (rankdata(col)-1)/max(len(col)-1,1), 0, A)
    y = np.array([l for _,l in chunks_labels])
    return A, R, y

# precompute per-date views (native + sim-live)
cache = {d: {"native": feats_for(bydate[d], False), "sim": feats_for(bydate[d], True)} for d in big}

def run(use_rel, test_mode):
    oof_s, oof_y = [], []
    for held in big:
        Xtr = np.vstack([np.hstack([cache[d]["native"][0], cache[d]["native"][1]]) if use_rel
                         else cache[d]["native"][0] for d in big if d != held])
        ytr = np.concatenate([cache[d]["native"][2] for d in big if d != held])
        A,R,y = cache[held][test_mode]
        Xte = np.hstack([A,R]) if use_rel else A
        m = lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xtr, ytr)
        oof_s.extend(m.predict_proba(Xte)[:,1]); oof_y.extend(y)
    s=np.array(oof_s); yy=np.array(oof_y)
    cal=calibrate(s,yy); sc=apply_calibration(s,cal); r,d=reward(sc,yy)
    return r, average_precision_score(yy,sc), d["bot_recall"]

for mode in ("native","sim"):
    print(f"\n=== test on {mode.upper()} ===")
    for use_rel,tag in [(False,"absolute only (309)"),(True,"absolute+relative (618)")]:
        r,ap,rec=run(use_rel,mode)
        print(f"  {tag:26s} reward={r:.4f} AP={ap:.4f} recall@5%FPR={rec:.4f}")
