"""Permutation-invariant sequence model for Poker44 chunk scoring.

A GRU reads each hand's action sequence in order (capturing within-hand action
order that aggregate features discard), producing a hand embedding; hands are
then pooled with mean+max — permutation-invariant across hands, so there is NO
hand-ordering leakage. Its probabilities are blended with the LightGBM model's;
the blend adds orthogonal signal (validated to improve both leave-date-out OOF
and the API-split holdout).

Torch is an optional dependency: if it is unavailable the caller falls back to
LightGBM-only scoring.
"""

from __future__ import annotations

import numpy as np

from poker44.model.features import _safe, _snap_bucket

MAX_H, MAX_A = 40, 14
_ATYPE = {
    "fold": 0, "call": 1, "check": 2, "bet": 3, "raise": 4,
    "small_blind": 5, "big_blind": 6, "ante": 7,
}
_STR = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
BLEND_SEQ_WEIGHT = 0.3  # validated sweet spot (improves both OOF and api-split)


def encode_chunk(chunk: list[dict]) -> tuple:
    """Encode a chunk (list of viewed hands) into fixed-shape arrays."""
    ai = np.full((MAX_H, MAX_A), 8, np.int64)
    si = np.full((MAX_H, MAX_A), 4, np.int64)
    ri = np.zeros((MAX_H, MAX_A), np.int64)
    bi = np.full((MAX_H, MAX_A), 16, np.int64)
    cont = np.zeros((MAX_H, MAX_A, 3), np.float32)
    amask = np.zeros((MAX_H, MAX_A), np.float32)
    hmask = np.zeros(MAX_H, np.float32)
    for h, hand in enumerate((chunk or [])[:MAX_H]):
        meta = hand.get("metadata") or {}
        hero = meta.get("hero_seat")
        bb = _safe(meta.get("bb"), 0.0) or 1.0
        acts = hand.get("actions") or []
        if acts:
            hmask[h] = 1.0
        for a, act in enumerate(acts[:MAX_A]):
            ai[h, a] = _ATYPE.get(act.get("action_type") or "", 8)
            si[h, a] = _STR.get(act.get("street") or "", 4)
            ri[h, a] = 1 if act.get("actor_seat") == hero else 0
            nbb = _safe(act.get("normalized_amount_bb"))
            bi[h, a] = _snap_bucket(nbb) if nbb > 0 else 16
            cont[h, a] = [
                nbb / 50.0,
                _safe(act.get("pot_before")) / bb / 100.0,
                _safe(act.get("pot_after")) / bb / 100.0,
            ]
            amask[h, a] = 1.0
    return ai, si, ri, bi, cont, amask, hmask


def _build_module(d: int = 32):
    import torch.nn as nn

    class SeqSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ea = nn.Embedding(9, 8)
            self.es = nn.Embedding(5, 4)
            self.er = nn.Embedding(2, 2)
            self.eb = nn.Embedding(17, 8)
            self.proj = nn.Linear(8 + 4 + 2 + 8 + 3, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, 1)
            )

        def forward(self, ai, si, ri, bi, cont, amask, hmask):
            import torch

            b = ai.shape[0]
            x = torch.cat(
                [self.ea(ai), self.es(si), self.er(ri), self.eb(bi), cont], -1
            )
            x = torch.relu(self.proj(x)).view(b * MAX_H, MAX_A, -1)
            lengths = amask.view(b * MAX_H, MAX_A).sum(1).clamp(min=1)
            out, _ = self.gru(x)
            idx = (lengths - 1).long().view(-1, 1, 1).expand(-1, 1, out.shape[-1])
            hand = out.gather(1, idx).squeeze(1).view(b, MAX_H, -1)
            hm = hmask.unsqueeze(-1)
            mean = (hand * hm).sum(1) / hm.sum(1).clamp(min=1)
            mx = (hand + (hm - 1) * 1e9).max(1).values
            return self.head(torch.cat([mean, mx], -1)).squeeze(-1)

    return SeqSet()


def train_sequence(enc_train, y_train, enc_val, y_val, *, seed: int = 44, max_epochs: int = 40):
    """Train the sequence model; early-stop on validation LOSS (not AP, which
    pegs at ~1.0 on the trivially-separable benchmark and freezes the net)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = _build_module()
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    def batch(enc_list, idxs):
        b = [enc_list[i] for i in idxs]
        return [torch.tensor(np.stack([x[k] for x in b])) for k in range(7)]

    n = len(y_train)
    best, best_state, patience = 1e9, None, 0
    rng = np.random.RandomState(seed)
    for _ in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, 128):
            gi = perm[s:s + 128].tolist()
            opt.zero_grad()
            out = model(*batch(enc_train, gi))
            loss = lossf(out, torch.tensor(y_train[gi], dtype=torch.float32))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(*batch(enc_val, list(range(len(y_val)))))
            vl = lossf(vout, torch.tensor(y_val, dtype=torch.float32)).item()
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_sequence(model, enc_list) -> np.ndarray:
    import torch

    if not enc_list:
        return np.zeros(0, dtype=float)
    with torch.no_grad():
        t = [torch.tensor(np.stack([x[k] for x in enc_list])) for k in range(7)]
        return torch.sigmoid(model(*t)).numpy()


class SequenceScorer:
    """Loads a saved sequence model for inference; None if torch/model absent."""

    def __init__(self, weights_path):
        self.model = None
        try:
            import torch

            model = _build_module()
            model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
            model.eval()
            self.model = model
        except Exception:  # noqa: BLE001
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def score(self, chunks: list[list[dict]]) -> np.ndarray:
        enc = [encode_chunk(c or []) for c in chunks]
        return predict_sequence(self.model, enc)
