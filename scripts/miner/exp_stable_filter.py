"""Validate the stable-feature filter using REAL benchmark->live drift.

Drop features that shift most benchmark->live, retrain, and evaluate at the
live chunk size (~90 hands, pooled) vs native. If dropping drifters improves
robustness at live size without collapsing native OOF, it should help live.
"""
import sys, json, glob, random
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.model.features import chunk_features
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from scripts.miner.train_model import LGB_PARAMS, calibrate, apply_calibration

# benchmark groups by (date,label)
groups = []
for dd in sorted(Path("/root/Poker44-subnet/data/benchmark").iterdir()):
    if not dd.is_dir(): continue
    for f in sorted(dd.glob("*.json")):
        if f.name=="manifest.json": continue
        p=json.loads(f.read_text())
        if len(p.get("chunks") or [])!=len(p.get("groundTruth") or []): continue
        for g,l in zip(p["chunks"],p["groundTruth"]):
            groups.append((dd.name,int(l),[prepare_hand_for_miner(h) for h in g]))
names=sorted({k for _,_,g in groups[:1] for k in chunk_features(g)})
# recompute full names robustly
allfeat=[chunk_features(g) for _,_,g in groups]
names=sorted({k for f in allfeat for k in f})
X=np.array([[f.get(k,0.0) for k in names] for f in allfeat])
y=np.array([l for _,l,_ in groups]); dates=np.array([d for d,_,_ in groups])

# live features
live=[]
for line in Path("/root/Poker44-subnet/data/live_chunks/live_chunks.jsonl").read_text().splitlines():
    if line.strip(): live.append(chunk_features(json.loads(line)["hands"]))
Xl=np.array([[f.get(k,0.0) for k in names] for f in live])

# robust drift: |mean_live-mean_bench| / (std_bench + |mean_bench|*0.1 + eps)
mu_b,sd_b=X.mean(0),X.std(0)
drift=np.abs(Xl.mean(0)-mu_b)/(sd_b+np.abs(mu_b)*0.1+1e-6)

ud=sorted(set(dates.tolist())); big=[d for d in ud if (dates==d).sum()>=50]
folds=[[d] for d in big]+[[d for d in ud if d not in big]]

def pooled(te_groups, rng, target=90):
    by={}
    for d,l,g in te_groups: by.setdefault((d,l),[]).append(g)
    Xp,yp=[],[]
    for (d,l),gl in by.items():
        for g in gl:
            acc=list(g)
            while len(acc)<target: acc+=list(gl[rng.randrange(len(gl))])
            Xp.append(chunk_features(acc[:target])); yp.append(l)
    return Xp,yp

def eval_keep(keep_mask):
    rng=random.Random(0)
    oofN=np.zeros(len(y)); yN=y.copy()
    poolS=[]; poolY=[]
    for fd in folds:
        te=np.isin(dates,fd); tr=~te
        m=lgb.LGBMClassifier(**LGB_PARAMS); m.fit(X[tr][:,keep_mask],y[tr])
        oofN[te]=m.predict_proba(X[te][:,keep_mask])[:,1]
        te_groups=[groups[i] for i in np.where(te)[0]]
        Xp,yp=pooled(te_groups,rng)
        Xp=np.array([[f.get(k,0.0) for k in names] for f in Xp])[:,keep_mask]
        poolS.extend(m.predict_proba(Xp)[:,1]); poolY.extend(yp)
    def sc(s,l):
        s=np.array(s); l=np.array(l); cal=calibrate(s,l)
        r,d=reward(apply_calibration(s,cal),l); return r,average_precision_score(l,s),d["bot_recall"]
    rN,apN,recN=sc(oofN,yN); rP,apP,recP=sc(poolS,poolY)
    return keep_mask.sum(),rN,recN,rP,recP

print(f"{len(groups)} bench, {len(live)} live chunks, {len(names)} features")
print(f"{'features':>8} | native OOF reward/recall | POOLED-90h reward/recall")
for thr in (np.inf, 3.0, 2.0, 1.5, 1.0):
    keep = drift<=thr if thr!=np.inf else np.ones(len(names),bool)
    n,rN,recN,rP,recP=eval_keep(keep)
    tag = "ALL" if thr==np.inf else f"drift<={thr}"
    print(f"  {tag:12s} {n:3d} feats | native {rN:.4f}/{recN:.4f} | POOLED-90h {rP:.4f}/{recP:.4f}")
