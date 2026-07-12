"""Validate a sequence model: GRU over actions within a hand (captures order),
permutation-invariant mean+max pool over hands (no hand-order leakage).
Blend its OOF probs with the LightGBM OOF probs and check if the blend beats
LGBM-only on BOTH pooled out-of-fold reward AND the api-split holdout.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.model.features import chunk_features, _snap_bucket, _safe
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from scripts.miner.train_model import LGB_PARAMS, calibrate, apply_calibration

torch.manual_seed(44)
MAX_H, MAX_A = 40, 14
ATYPE = {"fold":0,"call":1,"check":2,"bet":3,"raise":4,"small_blind":5,"big_blind":6,"ante":7}
STR = {"preflop":0,"flop":1,"turn":2,"river":3}

def encode(chunk):
    ai = np.full((MAX_H, MAX_A), 8, np.int64); si = np.full((MAX_H, MAX_A), 4, np.int64)
    ri = np.zeros((MAX_H, MAX_A), np.int64); bi = np.full((MAX_H, MAX_A), 16, np.int64)
    cont = np.zeros((MAX_H, MAX_A, 3), np.float32)
    amask = np.zeros((MAX_H, MAX_A), np.float32); hmask = np.zeros(MAX_H, np.float32)
    for h, hand in enumerate(chunk[:MAX_H]):
        meta = hand.get("metadata") or {}; hero = meta.get("hero_seat")
        bb = _safe(meta.get("bb"),0.0) or 1.0
        acts = hand.get("actions") or []
        if acts: hmask[h] = 1.0
        for a, act in enumerate(acts[:MAX_A]):
            ai[h,a] = ATYPE.get(act.get("action_type") or "", 8)
            si[h,a] = STR.get(act.get("street") or "", 4)
            ri[h,a] = 1 if act.get("actor_seat")==hero else 0
            nbb = _safe(act.get("normalized_amount_bb"))
            bi[h,a] = _snap_bucket(nbb) if nbb>0 else 16
            cont[h,a] = [nbb/50.0, _safe(act.get("pot_before"))/bb/100.0, _safe(act.get("pot_after"))/bb/100.0]
            amask[h,a] = 1.0
    return ai, si, ri, bi, cont, amask, hmask

class SeqSet(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ea=nn.Embedding(9,8); self.es=nn.Embedding(5,4); self.er=nn.Embedding(2,2); self.eb=nn.Embedding(17,8)
        self.proj=nn.Linear(8+4+2+8+3, d)
        self.gru=nn.GRU(d, d, batch_first=True)
        self.head=nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d,1))
    def forward(self, ai,si,ri,bi,cont,amask,hmask):
        B=ai.shape[0]
        x=torch.cat([self.ea(ai),self.es(si),self.er(ri),self.eb(bi),cont],-1)
        x=torch.relu(self.proj(x)).view(B*MAX_H, MAX_A, -1)
        lengths=amask.view(B*MAX_H,MAX_A).sum(1).clamp(min=1)
        out,_=self.gru(x)                                   # (B*H, A, d)
        idx=(lengths-1).long().view(-1,1,1).expand(-1,1,out.shape[-1])
        hand=out.gather(1, idx).squeeze(1).view(B, MAX_H, -1)  # last valid step per hand
        hm=hmask.unsqueeze(-1)
        mean=(hand*hm).sum(1)/hm.sum(1).clamp(min=1)
        mx=(hand+(hm-1)*1e9).max(1).values
        return self.head(torch.cat([mean,mx],-1)).squeeze(-1)

# ---- load data ----
rows=[]  # (date, label, split, encoded, feat_dict)
for dd in sorted(Path("/root/Poker44-subnet/data/benchmark").iterdir()):
    if not dd.is_dir(): continue
    for f in sorted(dd.glob("*.json")):
        if f.name=="manifest.json": continue
        p=json.loads(f.read_text())
        if len(p.get("chunks") or [])!=len(p.get("groundTruth") or []): continue
        for g,l in zip(p["chunks"], p["groundTruth"]):
            v=[prepare_hand_for_miner(h) for h in g]
            rows.append((dd.name, int(l), p.get("split") or "train", encode(v), chunk_features(v)))
dates=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows]); spl=np.array([r[2] for r in rows])
fnames=sorted({k for r in rows for k in r[4]})
Xtab=np.array([[r[4].get(k,0.0) for k in fnames] for r in rows])
enc=[r[3] for r in rows]
print(f"{len(rows)} chunks, {Xtab.shape[1]} tab feats")

def to_tensors(idxs):
    b=[enc[i] for i in idxs]
    return [torch.tensor(np.stack([x[k] for x in b])) for k in range(7)]

def train_seq(tr_idx, va_idx):
    model=SeqSet(); opt=torch.optim.Adam(model.parameters(),1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss()
    yt=torch.tensor(y[tr_idx],dtype=torch.float32)
    best=1e9; best_state=None; patience=0
    for ep in range(40):
        model.train(); perm=np.random.permutation(len(tr_idx))
        for s in range(0,len(perm),128):
            bi_=perm[s:s+128]; gi=[tr_idx[j] for j in bi_]
            t=to_tensors(gi); opt.zero_grad()
            out=model(*t); loss=lossf(out, torch.tensor(y[gi],dtype=torch.float32))
            loss.backward(); opt.step()
        # val loss (early stop on LOSS not AP — benchmark AP pegs at 1.0)
        model.eval()
        with torch.no_grad():
            vout=model(*to_tensors(va_idx)); vl=lossf(vout, torch.tensor(y[va_idx],dtype=torch.float32)).item()
        if vl<best-1e-4: best=vl; best_state={k:v.clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=5: break
    model.load_state_dict(best_state); model.eval()
    return model

ud=sorted(set(dates.tolist())); big=[d for d in ud if (dates==d).sum()>=50]
folds=[[d] for d in big]+[[d for d in ud if d not in big]]
oof_seq=np.zeros(len(y)); oof_lgb=np.zeros(len(y))
for fd in folds:
    te=np.where(np.isin(dates,fd))[0]; tr=np.where(~np.isin(dates,fd))[0]
    # inner val split for early stopping: last 20% of tr by index
    rng=np.random.RandomState(0); rng.shuffle(tr); cut=int(len(tr)*0.85)
    trn, val = tr[:cut], tr[cut:]
    model=train_seq(trn.tolist(), val.tolist())
    with torch.no_grad():
        oof_seq[te]=torch.sigmoid(model(*to_tensors(te.tolist()))).numpy()
    m=lgb.LGBMClassifier(**LGB_PARAMS); m.fit(Xtab[tr],y[tr]); oof_lgb[te]=m.predict_proba(Xtab[te])[:,1]

def score(p, tag, mask=None):
    idx=np.ones(len(y),bool) if mask is None else mask
    cal=calibrate(p[idx],y[idx]); sc=apply_calibration(p[idx],cal)
    r,d=reward(sc,y[idx])
    print(f"  {tag:22s} reward={r:.4f} AP={average_precision_score(y[idx],sc):.4f} recall@5%FPR={d['bot_recall']:.4f}")
    return r

va_mask = spl=="validation"
np.savez("/root/Poker44-subnet/data/seq_oof.npz", oof_seq=oof_seq, oof_lgb=oof_lgb, y=y, va_mask=va_mask)
print("\nBLEND WEIGHT SWEEP (pooled OOF | api-split):")
for w in (0.0,0.15,0.2,0.25,0.3,0.35,0.4):
    blend=w*oof_seq+(1-w)*oof_lgb
    r_oof=score(blend, f"seq={w} POOLED")
    r_api=score(blend, f"seq={w} API-SPLIT", va_mask)
    print()
