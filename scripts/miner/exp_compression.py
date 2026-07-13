"""Test compression-complexity features (rank-1's distinctive signal) stacked on
the current absolute+relative model. Measure native + simulated-live, leave-date-out.
"""
import sys, json, gzip, math, copy
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.model.features import chunk_features, _safe
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from scripts.miner.train_model import LGB_PARAMS, calibrate, apply_calibration

_TOK = {"fold":"f","call":"c","check":"k","bet":"b","raise":"r",
        "small_blind":"s","big_blind":"g","ante":"a"}

def compression_features(chunk):
    """Cross-hand redundancy via compression/complexity — bots replay lines."""
    per_hand, all_stream = [], []
    for hand in chunk:
        toks = [_TOK.get(a.get("action_type") or "", "?") for a in (hand.get("actions") or [])]
        per_hand.append("".join(toks)); all_stream.extend(toks)
    flat = "".join(all_stream)
    f = {}
    if flat:
        cg = len(gzip.compress(flat.encode()))
        pg = sum(len(gzip.compress(s.encode())) for s in per_hand if s) or 1
        f["cx_gzip_ratio"] = cg / pg                      # <1 => cross-hand redundancy (bot)
        # order-1 conditional entropy H(a_t|a_{t-1})
        bg = Counter(zip(flat[:-1], flat[1:])); ug = Counter(flat[:-1])
        H = 0.0
        for (a,b),n in bg.items():
            p_ab = n/max(sum(bg.values()),1); p_b_a = n/ug[a]
            H -= p_ab*math.log(p_b_a+1e-12)
        f["cx_entropy_rate"] = H
        f["cx_stream_len"] = float(len(flat))
    else:
        f["cx_gzip_ratio"]=0.0; f["cx_entropy_rate"]=0.0; f["cx_stream_len"]=0.0
    # pairwise Jaccard similarity of per-hand action multisets (repetition)
    sets=[set(s) for s in per_hand if s]
    if len(sets)>=2:
        sims=[]
        for i in range(min(len(sets),40)):
            for j in range(i+1,min(len(sets),40)):
                u=sets[i]|sets[j]; sims.append(len(sets[i]&sets[j])/len(u) if u else 0)
        f["cx_pair_jaccard"]=float(np.mean(sims)) if sims else 0.0
    else:
        f["cx_pair_jaccard"]=0.0
    # distinct-hand-pattern ratio via compression of the pattern sequence
    f["cx_distinct_ratio"]=len(set(per_hand))/max(len(per_hand),1)
    return f

def rescale_hand(hand, target=100.0):
    h=copy.deepcopy(hand); meta=h.get("metadata") or {}; bb=_safe(meta.get("bb"),0.0) or 1.0
    hero=meta.get("hero_seat"); hs=0.0
    for p in h.get("players") or []:
        if p.get("seat")==hero: hs=_safe(p.get("starting_stack"))/bb
    if hs<=1.0: return h
    sc=target/hs
    for p in h.get("players") or []:
        if p.get("starting_stack") is not None: p["starting_stack"]=_safe(p.get("starting_stack"))*sc
    for a in h.get("actions") or []:
        for k in ("amount","pot_before","pot_after","normalized_amount_bb"):
            if a.get(k) is not None: a[k]=_safe(a.get(k))*sc
    return h

bydate={}
for dd in sorted(Path("/root/Poker44-subnet/data/benchmark").iterdir()):
    if not dd.is_dir(): continue
    for fp in sorted(dd.glob("*.json")):
        if fp.name=="manifest.json": continue
        p=json.loads(fp.read_text())
        if len(p.get("chunks") or [])!=len(p.get("groundTruth") or []): continue
        for g,l in zip(p["chunks"],p["groundTruth"]):
            bydate.setdefault(dd.name,[]).append(([prepare_hand_for_miner(h) for h in g],int(l)))
big=[d for d in bydate if len(bydate[d])>=50]

names=None
def build(cl, rescale, use_cx):
    global names
    chunks=[([rescale_hand(h) for h in c] if rescale else c) for c,_ in cl]
    F=[]
    for c in chunks:
        d=dict(chunk_features(c))
        if use_cx: d.update(compression_features(c))
        F.append(d)
    if names is None or use_cx: names=sorted({k for x in F for k in x})
    A=np.array([[x.get(k,0.0) for k in names] for x in F])
    R=np.apply_along_axis(lambda col:(rankdata(col)-1)/max(len(col)-1,1),0,A)
    y=np.array([l for _,l in cl])
    return np.hstack([A,R]), y

def run(use_cx, mode):
    global names; names=None
    oof_s,oof_y=[],[]
    for held in big:
        parts=[build(bydate[d],False,use_cx) for d in big if d!=held]
        Xtr=np.vstack([p[0] for p in parts]); ytr=np.concatenate([p[1] for p in parts])
        Xte,yte=build(bydate[held], mode=="sim", use_cx)
        m=lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xtr,ytr)
        oof_s.extend(m.predict_proba(Xte)[:,1]); oof_y.extend(yte)
    s=np.array(oof_s); yy=np.array(oof_y); cal=calibrate(s,yy); sc=apply_calibration(s,cal); r,d=reward(sc,yy)
    return r, average_precision_score(yy,sc), d["bot_recall"]

for mode in ("native","sim"):
    print(f"\n=== {mode.upper()} ===")
    for use_cx,tag in [(False,"abs+relative (current)"),(True,"abs+relative+compression")]:
        r,ap,rec=run(use_cx,mode); print(f"  {tag:28s} reward={r:.4f} AP={ap:.4f} recall@5%FPR={rec:.4f}")
