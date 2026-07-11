"""Chunk-level feature extraction for the Poker44 bot-detection model.

A "chunk" is a list of hand payload dicts (30-40 hands from one tracked
player).  Features combine per-hand behavioral signals with cross-hand
consistency statistics: bots tend to reuse bet sizes, keep stable action
mixes, and show low variance across hands, while humans drift.

Only numpy is required so the miner can run this at inference time with no
model-framework dependency.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

MEANINGFUL_ACTIONS = ("fold", "call", "check", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# The validator quantizes normalized_amount_bb to this 16-edge grid. Snapping
# to it before building sizing signatures cancels the injected bucket noise so
# repeated bot bet-lines collapse to identical tuples.
_BB_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)

_EPS = 1e-9


def _snap_bucket(value: float) -> int:
    """Index of the nearest big-blind bucket edge (0..15)."""
    best_i, best_d = 0, float("inf")
    for i, edge in enumerate(_BB_BUCKETS):
        d = abs(value - edge)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _norm_entropy(values: list) -> float:
    """Shannon entropy of a sequence, normalized to [0,1] by log(#categories)."""
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    ent = -sum((c / total) * math.log((c / total) + 1e-12) for c in counts.values())
    return ent / math.log(len(counts))


def _max_run_share(values: list) -> float:
    """Longest run of an identical consecutive value / length (regularity tell)."""
    if not values:
        return 0.0
    longest = cur = 1
    for prev, nxt in zip(values, values[1:]):
        cur = cur + 1 if prev == nxt else 1
        longest = max(longest, cur)
    return longest / len(values)


def _safe(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _decimal_places(value: float) -> int:
    """Digits after the decimal point of a monetary amount (max 6)."""
    text = f"{value:.6f}".rstrip("0")
    if "." not in text:
        return 0
    return min(len(text.split(".", 1)[1]), 6)


def hand_features(hand: dict) -> dict:
    """Per-hand behavioral features. Tolerates missing fields."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}

    big_blind = _safe(metadata.get("bb"), 0.0) or 1.0
    hero_seat = metadata.get("hero_seat")

    feats: dict[str, float] = {}
    feats["n_players"] = float(len(players))
    feats["n_streets"] = float(len(streets))
    feats["n_actions"] = float(len(actions))
    feats["showdown"] = 1.0 if outcome.get("showdown") else 0.0
    feats["total_pot_bb"] = _safe(outcome.get("total_pot")) / big_blind

    stacks = [_safe(p.get("starting_stack")) / big_blind for p in players]
    feats["stack_mean_bb"] = float(np.mean(stacks)) if stacks else 0.0
    feats["stack_std_bb"] = float(np.std(stacks)) if stacks else 0.0

    counts = Counter((a.get("action_type") or "") for a in actions)
    meaningful = max(1, sum(counts.get(k, 0) for k in MEANINGFUL_ACTIONS))
    for kind in MEANINGFUL_ACTIONS:
        feats[f"{kind}_ratio"] = counts.get(kind, 0) / meaningful
    feats["aggression"] = (counts.get("bet", 0) + counts.get("raise", 0)) / meaningful

    street_counts = Counter((a.get("street") or "") for a in actions)
    n_act = max(1, len(actions))
    for street in STREETS:
        feats[f"{street}_share"] = street_counts.get(street, 0) / n_act

    amounts_bb = [
        _safe(a.get("normalized_amount_bb"))
        for a in actions
        if _safe(a.get("normalized_amount_bb")) > 0
    ]
    feats["amt_mean_bb"] = float(np.mean(amounts_bb)) if amounts_bb else 0.0
    feats["amt_std_bb"] = float(np.std(amounts_bb)) if amounts_bb else 0.0
    feats["amt_max_bb"] = float(np.max(amounts_bb)) if amounts_bb else 0.0

    pots_before = [_safe(a.get("pot_before")) / big_blind for a in actions]
    pots_after = [_safe(a.get("pot_after")) / big_blind for a in actions]
    feats["pot_before_mean"] = float(np.mean(pots_before)) if pots_before else 0.0
    feats["pot_after_max"] = float(np.max(pots_after)) if pots_after else 0.0

    # Bet sizing as a fraction of the pot for aggressive actions.
    pot_fracs = []
    precisions = []
    for a in actions:
        kind = a.get("action_type")
        amount = _safe(a.get("amount"))
        if amount > 0:
            precisions.append(float(_decimal_places(amount)))
        if kind in ("bet", "raise"):
            pot = _safe(a.get("pot_before"))
            if pot > _EPS and amount > 0:
                pot_fracs.append(amount / pot)
    feats["potfrac_mean"] = float(np.mean(pot_fracs)) if pot_fracs else 0.0
    feats["potfrac_std"] = float(np.std(pot_fracs)) if pot_fracs else 0.0
    feats["amt_precision_mean"] = float(np.mean(precisions)) if precisions else 0.0

    # Hero-specific behavior: the tracked entity's own action mix and sizing.
    hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat]
    hero_counts = Counter((a.get("action_type") or "") for a in hero_actions)
    hero_meaningful = max(1, sum(hero_counts.get(k, 0) for k in MEANINGFUL_ACTIONS))
    feats["hero_n_actions"] = float(len(hero_actions))
    for kind in MEANINGFUL_ACTIONS:
        feats[f"hero_{kind}_ratio"] = hero_counts.get(kind, 0) / hero_meaningful
    feats["hero_aggression"] = (
        hero_counts.get("bet", 0) + hero_counts.get("raise", 0)
    ) / hero_meaningful

    hero_amounts = [
        _safe(a.get("normalized_amount_bb"))
        for a in hero_actions
        if _safe(a.get("normalized_amount_bb")) > 0
    ]
    feats["hero_amt_mean_bb"] = float(np.mean(hero_amounts)) if hero_amounts else 0.0
    feats["hero_amt_std_bb"] = float(np.std(hero_amounts)) if hero_amounts else 0.0

    hero_fracs = []
    for a in hero_actions:
        if a.get("action_type") in ("bet", "raise"):
            pot = _safe(a.get("pot_before"))
            amount = _safe(a.get("amount"))
            if pot > _EPS and amount > 0:
                hero_fracs.append(amount / pot)
    feats["hero_potfrac_mean"] = float(np.mean(hero_fracs)) if hero_fracs else 0.0
    feats["hero_potfrac_std"] = float(np.std(hero_fracs)) if hero_fracs else 0.0

    # Hero position relative to the button (bots may ignore position).
    button_seat = metadata.get("button_seat")
    if isinstance(hero_seat, int) and isinstance(button_seat, int) and players:
        feats["hero_btn_offset"] = float((hero_seat - button_seat) % max(len(players), 1))
    else:
        feats["hero_btn_offset"] = -1.0

    # Per-street hero involvement and aggression.
    for street in STREETS:
        street_hero = [a for a in hero_actions if a.get("street") == street]
        n_street = max(1, len(street_hero))
        agg = sum(1 for a in street_hero if a.get("action_type") in ("bet", "raise"))
        feats[f"hero_{street}_n"] = float(len(street_hero))
        feats[f"hero_{street}_agg"] = agg / n_street

    # Hero voluntarily-put-money-in-pot proxy (preflop call/bet/raise).
    preflop_hero = [a for a in hero_actions if a.get("street") == "preflop"]
    feats["hero_vpip"] = (
        1.0
        if any(a.get("action_type") in ("call", "bet", "raise") for a in preflop_hero)
        else 0.0
    )
    feats["hero_showed"] = 0.0
    for p in players:
        if p.get("seat") == hero_seat and p.get("showed_hand"):
            feats["hero_showed"] = 1.0

    # ---- rank-1-derived structural signals (composition, not order) ----
    action_types = [a.get("action_type") or "" for a in actions]
    actor_seats = [a.get("actor_seat") for a in actions if a.get("actor_seat")]
    street_names = [a.get("street") or "" for a in actions]
    n_act = max(1, len(actions))

    feats["action_entropy"] = _norm_entropy(action_types)
    feats["actor_entropy"] = _norm_entropy(actor_seats)
    feats["street_entropy"] = _norm_entropy(street_names)
    feats["action_run_max_share"] = _max_run_share(action_types)
    feats["actor_run_max_share"] = _max_run_share(actor_seats)
    feats["actor_switch_rate"] = (
        sum(1 for p, n in zip(actor_seats, actor_seats[1:]) if p != n)
        / max(1, len(actor_seats) - 1)
    )
    feats["unique_actor_share"] = len(set(actor_seats)) / max(1, len(players))
    feats["seat_utilization"] = len(players) / max(1, _safe(metadata.get("max_seats"), 6))

    pots_after_seq = [_safe(a.get("pot_after")) for a in actions]
    feats["pot_monotonic_rate"] = (
        sum(1 for p, n in zip(pots_after_seq, pots_after_seq[1:]) if n + 1e-9 >= p)
        / max(1, len(pots_after_seq) - 1)
    )
    if pots_after_seq and pots_before:
        feats["pot_growth_bb"] = max(pots_after_seq) - min(pots_before)
    else:
        feats["pot_growth_bb"] = 0.0

    feats["raise_to_share"] = sum(1 for a in actions if a.get("raise_to") is not None) / n_act
    feats["call_to_share"] = sum(1 for a in actions if a.get("call_to") is not None) / n_act
    feats["nonzero_amount_share"] = sum(1 for a in actions if _safe(a.get("normalized_amount_bb")) > 0) / n_act

    button_seat = metadata.get("button_seat")
    feats["button_action_share"] = (
        sum(1 for s in actor_seats if s == button_seat) / n_act
        if isinstance(button_seat, int)
        else 0.0
    )
    feats["hero_button_same"] = 1.0 if (hero_seat and hero_seat == button_seat) else 0.0

    return feats


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([c / total for c in counter.values() if c > 0])
    return float(-np.sum(probs * np.log2(probs)))


def _seq_stats(tokens: list, prefix: str, feats: dict) -> None:
    """Unigram/bigram regularity of a token stream.

    Bots produce low-entropy, highly-repetitive action streams; humans are
    noisier. Works on the hero's action stream pooled across a whole chunk,
    since per-hand hero actions are too sparse (the served window shows only
    5-8 table actions, so the hero often acts 0-1 times per hand).
    """
    n = len(tokens)
    feats[f"{prefix}_len"] = float(n)
    uni = Counter(tokens)
    feats[f"{prefix}_uni_entropy"] = _entropy(uni)
    feats[f"{prefix}_uni_distinct"] = float(len(uni))
    bigrams = Counter(zip(tokens[:-1], tokens[1:])) if n >= 2 else Counter()
    total_bi = sum(bigrams.values())
    if total_bi > 0:
        feats[f"{prefix}_bi_entropy"] = _entropy(bigrams)
        feats[f"{prefix}_bi_top_share"] = max(bigrams.values()) / total_bi
        feats[f"{prefix}_bi_distinct_ratio"] = len(bigrams) / total_bi
    else:
        feats[f"{prefix}_bi_entropy"] = 0.0
        feats[f"{prefix}_bi_top_share"] = 0.0
        feats[f"{prefix}_bi_distinct_ratio"] = 0.0


def chunk_features(chunk: list[dict]) -> dict:
    """Aggregate hand features + cross-hand consistency stats for one chunk."""
    feats: dict[str, float] = {"n_hands": float(len(chunk or []))}
    if not chunk:
        return feats

    per_hand = [hand_features(h) for h in chunk]
    keys = sorted(per_hand[0].keys())
    for key in keys:
        values = np.asarray([h.get(key, 0.0) for h in per_hand], dtype=float)
        feats[f"{key}__mean"] = float(np.mean(values))
        feats[f"{key}__std"] = float(np.std(values))
        feats[f"{key}__min"] = float(np.min(values))
        feats[f"{key}__max"] = float(np.max(values))

    # Cross-hand consistency of the hero's bet sizing: bots reuse exact sizes.
    hero_amounts: list[float] = []
    hero_action_counter: Counter = Counter()
    all_amount_counter: Counter = Counter()
    for hand in chunk:
        metadata = hand.get("metadata") or {}
        hero_seat = metadata.get("hero_seat")
        for action in hand.get("actions") or []:
            amt = _safe(action.get("normalized_amount_bb"))
            if amt > 0:
                all_amount_counter[round(amt, 2)] += 1
            if action.get("actor_seat") == hero_seat:
                hero_action_counter[action.get("action_type") or ""] += 1
                if amt > 0 and action.get("action_type") in ("bet", "raise", "call"):
                    hero_amounts.append(amt)

    n_hero_amounts = len(hero_amounts)
    distinct = len({round(a, 2) for a in hero_amounts})
    feats["hero_amt_uniqueness"] = distinct / n_hero_amounts if n_hero_amounts else 0.0
    feats["hero_amt_count"] = float(n_hero_amounts)
    feats["hero_action_entropy"] = _entropy(hero_action_counter)
    feats["table_amt_entropy"] = _entropy(all_amount_counter)
    if hero_amounts:
        arr = np.asarray(hero_amounts)
        feats["hero_amt_cv"] = float(np.std(arr) / (np.mean(arr) + _EPS))
        feats["hero_amt_p90"] = float(np.percentile(arr, 90))
    else:
        feats["hero_amt_cv"] = 0.0
        feats["hero_amt_p90"] = 0.0

    # Histogram of the hero's aggressive sizing as pot fractions: bots often
    # concentrate on canonical sizes (1/3, 1/2, 2/3, 3/4, pot, overbet).
    frac_edges = [0.0, 0.4, 0.6, 0.8, 1.05, np.inf]
    frac_counts = np.zeros(len(frac_edges) - 1)
    exact_hits = 0
    hero_frac_list: list[float] = []
    for hand in chunk:
        metadata = hand.get("metadata") or {}
        hero_seat = metadata.get("hero_seat")
        for action in hand.get("actions") or []:
            if action.get("actor_seat") != hero_seat:
                continue
            if action.get("action_type") not in ("bet", "raise"):
                continue
            pot = _safe(action.get("pot_before"))
            amount = _safe(action.get("amount"))
            if pot > _EPS and amount > 0:
                frac = amount / pot
                hero_frac_list.append(frac)
                for i in range(len(frac_edges) - 1):
                    if frac_edges[i] <= frac < frac_edges[i + 1]:
                        frac_counts[i] += 1
                        break
                for canonical in (0.33, 0.5, 0.66, 0.75, 1.0):
                    if abs(frac - canonical) < 0.02:
                        exact_hits += 1
                        break
    total_fracs = max(1, len(hero_frac_list))
    for i in range(len(frac_counts)):
        feats[f"hero_fracbin_{i}"] = float(frac_counts[i]) / total_fracs
    feats["hero_frac_canonical"] = exact_hits / total_fracs
    if hero_frac_list:
        arr = np.asarray(hero_frac_list)
        feats["hero_frac_group_std"] = float(np.std(arr))
        distinct_fracs = len({round(f, 2) for f in hero_frac_list})
        feats["hero_frac_uniqueness"] = distinct_fracs / total_fracs
    else:
        feats["hero_frac_group_std"] = 0.0
        feats["hero_frac_uniqueness"] = 0.0

    # ---- Sequence / cross-hand structure features ----
    # The hero's action stream pooled across the chunk (in hand then action
    # order), the hero's per-hand involvement, and how often whole-hand action
    # structures repeat across hands (a strong scripted-bot tell).
    hero_stream: list[str] = []
    hero_per_hand_counts: list[int] = []
    hand_patterns: list[tuple] = []
    hero_streets_reached: list[int] = []
    for hand in chunk:
        metadata = hand.get("metadata") or {}
        hero_seat = metadata.get("hero_seat")
        actions = hand.get("actions") or []
        hero_seq = [
            a.get("action_type") or "" for a in actions if a.get("actor_seat") == hero_seat
        ]
        hero_stream.extend(hero_seq)
        hero_per_hand_counts.append(len(hero_seq))
        hand_patterns.append(tuple((a.get("action_type") or "")[:2] for a in actions))
        hero_street_idx = [
            STREETS.index(a.get("street"))
            for a in actions
            if a.get("actor_seat") == hero_seat and a.get("street") in STREETS
        ]
        hero_streets_reached.append(max(hero_street_idx) if hero_street_idx else -1)

    _seq_stats(hero_stream, "hero_seq", feats)

    counts = np.asarray(hero_per_hand_counts, dtype=float)
    feats["hero_involve_mean"] = float(counts.mean()) if counts.size else 0.0
    feats["hero_involve_std"] = float(counts.std()) if counts.size else 0.0
    feats["hero_visible_frac"] = float((counts > 0).mean()) if counts.size else 0.0

    reached = np.asarray(hero_streets_reached, dtype=float)
    visible = reached[reached >= 0]
    feats["hero_reach_mean"] = float(visible.mean()) if visible.size else -1.0
    feats["hero_reach_std"] = float(visible.std()) if visible.size else 0.0

    pat_counter = Counter(hand_patterns)
    n_hands = max(1, len(hand_patterns))
    feats["handpat_distinct_ratio"] = len(pat_counter) / n_hands
    feats["handpat_top_share"] = (
        max(pat_counter.values()) / n_hands if pat_counter else 0.0
    )
    feats["handpat_entropy"] = _entropy(pat_counter)
    feats["handpat_repeat_frac"] = (
        float(sum(c for c in pat_counter.values() if c > 1)) / n_hands
    )

    # ---- Signature-replay features (scripted bots repeat exact lines) ----
    # Per-hand tuple signatures over four projections; role tokens (hero/other)
    # survive the validator's seat re-aliasing, bucket-snapped amounts survive
    # its amount noise. High top_share / low unique_share => replayed lines.
    action_sigs, role_sigs, street_sigs, amt_sigs = [], [], [], []
    for hand in chunk:
        metadata = hand.get("metadata") or {}
        hero_seat = metadata.get("hero_seat")
        actions = hand.get("actions") or []
        action_sigs.append(tuple((a.get("action_type") or "")[:2] for a in actions))
        role_sigs.append(
            tuple("H" if a.get("actor_seat") == hero_seat else "o" for a in actions)
        )
        street_sigs.append(tuple((a.get("street") or "")[:2] for a in actions))
        amt_sigs.append(
            tuple(_snap_bucket(_safe(a.get("normalized_amount_bb"))) for a in actions)
        )
    for name, sigs in (
        ("actsig", action_sigs),
        ("rolesig", role_sigs),
        ("streetsig", street_sigs),
        ("amtsig", amt_sigs),
    ):
        counter = Counter(sigs)
        denom = max(1, len(sigs))
        feats[f"{name}_top_share"] = max(counter.values()) / denom if counter else 0.0
        feats[f"{name}_unique_share"] = len(counter) / denom if counter else 0.0
        feats[f"{name}_entropy"] = _entropy(counter)

    # ---- Chunk-pooled per-street action & aggression shares ----
    # Bots fold preflop more but over-commit later streets; humans taper.
    street_action = Counter()
    street_aggro = Counter()
    total_street_actions = 0
    for hand in chunk:
        for action in hand.get("actions") or []:
            street = action.get("street")
            if street in _STREET_ORDER:
                street_action[street] += 1
                total_street_actions += 1
                if action.get("action_type") in ("bet", "raise"):
                    street_aggro[street] += 1
    for street in STREETS:
        feats[f"pooled_{street}_action_share"] = (
            street_action[street] / max(1, total_street_actions)
        )
        feats[f"pooled_{street}_aggro_share"] = (
            street_aggro[street] / street_action[street]
            if street_action[street]
            else 0.0
        )

    # ---- Poker-validity anomalies as bot tells (orthogonal to behavior) ----
    acted_after_fold = street_regression = street_jump = 0
    zero_amt_betraise = pot_mismatch = total_anom_actions = 0
    for hand in chunk:
        actions = hand.get("actions") or []
        folded_seats: set = set()
        prev_idx = -1
        for action in actions:
            total_anom_actions += 1
            seat = action.get("actor_seat")
            kind = action.get("action_type")
            idx = _STREET_ORDER.get(action.get("street"), -1)
            if seat in folded_seats:
                acted_after_fold += 1
            if kind == "fold":
                folded_seats.add(seat)
            if idx >= 0 and prev_idx >= 0:
                if idx < prev_idx:
                    street_regression += 1
                elif idx - prev_idx > 1:
                    street_jump += 1
            if idx >= 0:
                prev_idx = idx
            if kind in ("bet", "raise") and _safe(action.get("normalized_amount_bb")) <= 0:
                zero_amt_betraise += 1
        for i in range(len(actions) - 1):
            if abs(
                _safe(actions[i].get("pot_after"))
                - _safe(actions[i + 1].get("pot_before"))
            ) > 0.01:
                pot_mismatch += 1
    denom = max(1, total_anom_actions)
    feats["anom_acted_after_fold"] = acted_after_fold / denom
    feats["anom_street_regression"] = street_regression / denom
    feats["anom_street_jump"] = street_jump / denom
    feats["anom_zero_betraise"] = zero_amt_betraise / denom
    feats["anom_pot_mismatch"] = pot_mismatch / denom
    feats["chunk_anomaly_load"] = (
        acted_after_fold + street_regression + street_jump + zero_amt_betraise + pot_mismatch
    ) / denom

    # ---- Threshold-crossing hand-rate features (rank-1 nonlinear indicators) ----
    # Fraction of hands whose per-hand stat crosses a hard threshold — cheap
    # pre-binned signals trees exploit well. Composition-based, not order-based.
    n_hands_f = max(1, len(per_hand))
    feats["rate_high_aggression"] = (
        sum(1 for h in per_hand if h.get("aggression", 0.0) >= 0.35) / n_hands_f
    )
    feats["rate_low_action_entropy"] = (
        sum(1 for h in per_hand if h.get("action_entropy", 0.0) <= 0.35) / n_hands_f
    )
    feats["rate_high_actor_entropy"] = (
        sum(1 for h in per_hand if h.get("actor_entropy", 0.0) >= 0.75) / n_hands_f
    )
    feats["rate_long_action_hand"] = (
        sum(1 for h in per_hand if h.get("n_actions", 0.0) >= 12.0) / n_hands_f
    )
    feats["rate_high_monotonic"] = (
        sum(1 for h in per_hand if h.get("pot_monotonic_rate", 0.0) >= 0.99) / n_hands_f
    )

    # NOTE: temporal/session-drift features (autocorrelation, trend slope,
    # quartile drift over hand order) were tested and REJECTED — they exploit
    # the benchmark's hand-packing order (single-feature AP collapses when the
    # order is shuffled), pushing combined AP to 0.99, far above the subnet's
    # audit ceiling (0.76). That artifact won't exist in live eval, so those
    # features would fail in production. Do not re-add order-dependent features.

    return feats


def feature_vector(chunk: list[dict], feature_names: list[str]) -> np.ndarray:
    """Fixed-order numeric vector for model inference."""
    feats = chunk_features(chunk)
    return np.asarray([feats.get(name, 0.0) for name in feature_names], dtype=float)
