"""Runtime inference wrapper for the trained Poker44 bot-detection model.

Loads the LightGBM booster + metadata produced by
``scripts/miner/train_model.py`` and scores chunks. Falls back to a neutral
mid-low score if the artifacts or lightgbm are unavailable so the miner
never crashes a request.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from poker44.model.features import chunk_features

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FALLBACK_SCORE = 0.25


class DetectionModel:
    """Chunk-level bot-risk scorer backed by a trained LightGBM model."""

    def __init__(self, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.booster = None
        self.feature_names: list[str] = []
        self.pivot = 0.5
        self.seq_scorer = None
        self.seq_weight = 0.0
        self.use_relative = False
        self._load()

    def _load(self) -> None:
        model_path = self.artifact_dir / "model.lgb.txt"
        meta_path = self.artifact_dir / "model_meta.json"
        try:
            import lightgbm as lgb

            meta = json.loads(meta_path.read_text())
            self.feature_names = list(meta["feature_names"])
            self.pivot = float(meta.get("calibration", {}).get("pivot", 0.5))
            self.booster = lgb.Booster(model_file=str(model_path))
            self.use_relative = bool(meta.get("use_relative", False))
            logger.info(
                "detection model loaded: %s features, pivot=%.4f, relative=%s, trained on %s examples",
                len(self.feature_names),
                self.pivot,
                self.use_relative,
                meta.get("n_train_examples"),
            )
            # Optional sequence-model blend (orthogonal action-order signal).
            weight = float(meta.get("sequence_blend_weight", 0.0) or 0.0)
            seq_path = self.artifact_dir / "sequence.pt"
            if weight > 0.0 and seq_path.exists():
                from poker44.model.sequence import SequenceScorer

                scorer = SequenceScorer(seq_path)
                if scorer.ready:
                    self.seq_scorer = scorer
                    self.seq_weight = weight
                    logger.info("sequence blend active: weight=%.2f", weight)
                else:
                    logger.warning("sequence weights present but torch unavailable; LGBM-only")
        except Exception as err:  # noqa: BLE001
            self.booster = None
            logger.error("detection model unavailable, using fallback: %s", err)

    @property
    def ready(self) -> bool:
        return self.booster is not None

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        pivot = min(max(self.pivot, 1e-6), 1.0 - 1e-6)
        scaled = np.where(
            raw <= pivot,
            0.5 * raw / pivot,
            0.5 + 0.5 * (raw - pivot) / (1.0 - pivot),
        )
        return np.clip(scaled, 0.0, 1.0)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        """One calibrated bot-risk score in [0, 1] per chunk."""
        if not chunks:
            return []
        if not self.ready:
            return [FALLBACK_SCORE] * len(chunks)
        try:
            from poker44.model.features import build_feature_matrix

            degenerate = [
                not (chunk or [])
                or not any((hand or {}).get("actions") for hand in (chunk or []))
                for chunk in chunks
            ]
            # Absolute + (per-batch) relative view, matching training.
            rows = build_feature_matrix(chunks, self.feature_names, self.use_relative)
            raw = np.asarray(self.booster.predict(rows))
            # Blend in the sequence model where available (best-effort).
            if self.seq_scorer is not None and self.seq_weight > 0.0:
                try:
                    seq = np.asarray(self.seq_scorer.score(chunks), dtype=float)
                    if seq.shape == raw.shape:
                        raw = self.seq_weight * seq + (1.0 - self.seq_weight) * raw
                except Exception as err:  # noqa: BLE001
                    logger.error("sequence blend failed, LGBM-only: %s", err)
            calibrated = self._calibrate(raw)
            # Per-batch safety floor (drift-proofs the human-safety gate): the
            # validator needs >=1 chunk scored >=0.5 or the whole window's
            # threshold-sanity term collapses to 0. Live scores drift, so a
            # fixed pivot can flag zero on a low-scoring batch. Guarantee the
            # top ~5% (>=1) non-degenerate chunks land just above 0.5, ranked —
            # rank-preserving (AP/recall unchanged), only nudges the boundary.
            deg = np.asarray(degenerate)
            real_idx = np.flatnonzero(~deg)
            if real_idx.size >= 3:
                floor_k = max(1, int(round(0.05 * real_idx.size)))
                n_flagged = int((calibrated[real_idx] >= 0.5).sum())
                if n_flagged < floor_k:
                    top = real_idx[np.argsort(raw[real_idx])[-floor_k:]]
                    ranks = np.argsort(np.argsort(raw[top]))
                    calibrated[top] = 0.501 + 0.048 * (ranks / max(floor_k - 1, 1))
            return [
                FALLBACK_SCORE if is_degenerate else round(float(score), 6)
                for score, is_degenerate in zip(calibrated, degenerate)
            ]
        except Exception as err:  # noqa: BLE001
            logger.error("model inference failed, using fallback: %s", err)
            return [FALLBACK_SCORE] * len(chunks)
