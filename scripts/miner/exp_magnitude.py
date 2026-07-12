"""Does hero-stack magnitude normalization help under the CONFIRMED live shift?

Live stacks are ~100bb; benchmark ~230bb (10x larger pots). Simulate the shift:
rescale each held-out benchmark hand so hero=100bb (like live), then test a
model WITH vs WITHOUT hero-normalized magnitude features. If hero-norm is more
robust to the magnitude shift, it should win on the simulated-live eval.
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.model.features import chunk_features, _safe
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from scripts.miner.train_model import LGB_PARAMS, calibrate, apply_calibration

def rescale_hand(hand, target=100.0):
    """Rescale magnitudes so hero starting stack = target bb (simulate live)."""
    h = copy.deepcopy(hand)
    meta = h.get("metadata") or {}
    bb = _safe(meta.get("bb"), 0.0) or 1.0
    hero = meta.get("hero_seat")
    hs = 0.0
    for p in h.get("players") or []:
        if p.get("seat") == hero:
            hs = _safe(p.get("starting_stack")) / bb
    if hs <= 1.0:
        return h
    sc = target / hs
    for p in h.get("players") or []:
        if p.get("starting_stack") is not None:
            p["starting_stack"] = _safe(p.get("starting_stack")) * sc
    for a in h.get("actions") or []:
        for k in ("amount", "pot_before", "pot_after"):
            if a.get(k) is not None:
                a[k] = _safe(a.get(k)) * sc
        if a.get("normalized_amount_bb") is not None:
            a["normalized_amount_bb"] = _safe(a.get("normalized_amount_bb")) * sc
    return h

def hero_norm_feats(chunk):
    """chunk_features + hero-stack-normalized magnitude variants per chunk."""
    f = dict(chunk_features(chunk))
    # add chunk-level normalized magnitudes using median hero stack
    return f

# load benchmark
groups=[]
for dd in sorted(Path("/root/Poker44-subnet/data/benchmark").iterdir()):
    if not dd.is_dir(): continue
    for p in sorted(dd.glob("*.json")):
        if p.name=="manifest.json": continue
        pl=json.loads(p.read_text())
        if len(pl.get("chunks") or [])!=len(pl.get("groundTruth") or []): continue
        for g,l in zip(pl["chunks"],pl["groundTruth"]):
            groups.append((dd.name,int(l),[prepare_hand_for_miner(h) for h in g]))
y=np.array([l for _,l,_ in groups]); dates=np.array([d for d,_,_ in groups])
ud=sorted(set(dates.tolist())); big=[d for d in ud if (dates==d).sum()>=50]
folds=[[d] for d in big]+[[d for d in ud if d not in big]]

# Two feature variants: baseline (raw) and hero-normalized.
def feats_baseline(chunk): return chunk_features(chunk)
def feats_heronorm(chunk):
    f=dict(chunk_features(chunk))
    # per-chunk median hero stack -> scale magnitudes into hero=100bb units
    hs=[]
    for hand in chunk:
        m=hand.get("metadata") or {}; bb=_safe(m.get("bb"),0.0) or 1.0; hero=m.get("hero_seat")
        for p in hand.get("players") or []:
            if p.get("seat")==hero: hs.append(_safe(p.get("starting_stack"))/bb)
    sc = 100.0/np.median(hs) if hs and np.median(hs)>1 else 1.0
    sc = min(max(sc,0.1),10.0)
    for k in ("pot_before_mean__mean","pot_after_max__max","amt_max_bb__max","amt_mean_bb__mean",
              "pot_before_mean__max","total_pot_bb__mean"):
        if k in f: f[k+"_hn"]=f[k]*sc
    f["chunk_hero_stack_scale"]=sc
    return f

def build(gs, featfn, names=None):
    F=[featfn(g) for _,_,g in gs]
    if names is None: names=sorted({k for x in F for k in x})
    return np.array([[x.get(k,0.0) for k in names] for x in F]), names

def run(featfn, label):
    oof=np.zeros(len(y)); names=None
    for fd in folds:
        te=np.isin(dates,fd); tr=~te
        tr_g=[groups[i] for i in np.where(tr)[0]]
        Xtr,names=build(tr_g,featfn,names)
        m=lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xtr,y[tr])
        # simulated-live test fold: rescale each hand to hero=100bb
        te_g=[(d,l,[rescale_hand(h) for h in g]) for d,l,g in (groups[i] for i in np.where(te)[0])]
        Xte,_=build(te_g,featfn,names)
        oof[te]=m.predict_proba(Xte)[:,1]
    cal=calibrate(oof,y); sc=apply_calibration(oof,cal); r,d=reward(sc,y)
    print(f"  {label:28s} sim-live reward={r:.4f} AP={average_precision_score(y,sc):.4f} recall@5%FPR={d['bot_recall']:.4f}")

print("Evaluated on SIMULATED-LIVE (benchmark rescaled to hero=100bb):")
run(feats_baseline,"baseline (raw magnitudes)")
run(feats_heronorm,"+ hero-stack normalized")
