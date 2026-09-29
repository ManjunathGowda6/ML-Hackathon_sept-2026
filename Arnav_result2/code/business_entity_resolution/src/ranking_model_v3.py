"""
ML Challenge 2026 - Business Entity Resolution Pipeline
V3 Ranking & Adaptive Entity Decision Engine (Sections 16-23)

Key Capabilities:
1. Supervised Pairwise Classifier with HistGradientBoosting
2. Dynamic Evidence-Driven Adaptive K (K in [0, 8] conditioned on cluster density & probability slopes)
3. Contextual Singleton Confidence Gate (distinguishes true singletons from ambiguous false merges)
4. Source-Specific Calibration (S2 missing-address vs S3 transliteration resilience)
5. Multi-Channel Evidence Weighting (candidates supported by >= 2 channels receive confidence boost)
6. Contextual Score Margin & Slope Drop Termination
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class AdaptiveRankingModelV3:
    """
    V3 Entity-Level Decision & Ranking Model.
    """

    def __init__(
        self,
        base_match_threshold: float = 0.58,
        s2_missing_addr_threshold: float = 0.75,
        s3_translit_threshold: float = 0.38,
        base_margin: float = 0.12,
        tight_margin: float = 0.08,
        singleton_gate_base: float = 0.50,
        max_k_default: int = 5,
        max_k_ceiling: int = 8,
    ):
        self.base_match_threshold = base_match_threshold
        self.s2_missing_addr_threshold = s2_missing_addr_threshold
        self.s3_translit_threshold = s3_translit_threshold
        self.base_margin = base_margin
        self.tight_margin = tight_margin
        self.singleton_gate_base = singleton_gate_base
        self.max_k_default = max_k_default
        self.max_k_ceiling = max_k_ceiling

        self.classifier = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=50,
            max_depth=8,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        print(f"Training V3 Ranking Classifier on {len(X):,} candidate pairs...", flush=True)
        self.classifier.fit(X, y, sample_weight=sample_weight)
        print("V3 Classifier trained successfully.", flush=True)

    def decide_entity_matches(
        self,
        cand_ids: List[str],
        probs: np.ndarray,
        cand_features: np.ndarray,
    ) -> List[str]:
        """
        Decides the exact set of matched entity IDs for a single Source 1 entity
        using evidence-driven dynamic K, margin gating, and contextual singleton detection.
        """
        if len(probs) == 0:
            return []

        ranked_indices = np.argsort(-probs)
        best_idx = ranked_indices[0]
        best_prob = probs[best_idx]
        best_feat = cand_features[best_idx]

        # Feature indices in features_v3:
        # 12: generic_only_match, 14: both_have_address, 17: addr_joint_match
        # 25: name_high_addr_conflict, 27: retrieval_num_channels, 29: transliteration_candidate
        # 30: is_source2, 31: is_source3
        generic_only = best_feat[12] == 1.0
        both_have_addr = best_feat[14] == 1.0
        addr_joint = best_feat[17] == 1.0
        name_conflict = best_feat[25] == 1.0
        num_channels = best_feat[27]
        is_translit = best_feat[29] == 1.0

        # 1. Contextual Singleton Gate (Section 19)
        eff_singleton_gate = self.singleton_gate_base
        if addr_joint and both_have_addr:
            eff_singleton_gate = self.s3_translit_threshold
        elif generic_only and not both_have_addr:
            eff_singleton_gate = 0.65
        elif name_conflict:
            eff_singleton_gate = 0.65
        elif num_channels >= 3:
            eff_singleton_gate = max(self.singleton_gate_base - 0.05, 0.42)

        if best_prob < eff_singleton_gate:
            return []

        # 2. Dynamic K Determination (Sections 16 & 17)
        # Ground truth has 11.4% K in [6, 8]. Allow K up to 8 only if:
        # - Top candidate has strong probability (>= 0.70)
        # - Entity shows dense cluster of strong candidates (at least 5 candidates with prob >= 0.55)
        # Otherwise use conservative default K <= 5.
        # If top candidate is borderline (prob < 0.60), restrict to K <= 2 to prevent distractor flooding.
        num_strong = sum(1 for p in probs if p >= 0.55)
        if best_prob >= 0.70 and num_strong >= 5:
            effective_max_k = self.max_k_ceiling
        elif best_prob < 0.60:
            effective_max_k = min(self.max_k_default, 2)
        else:
            effective_max_k = self.max_k_default

        # 3. Dynamic Margin Determination (Section 23)
        eff_margin = self.base_margin if best_prob >= 0.65 else self.tight_margin

        # 4. Candidate Selection
        selected = []
        for rank, idx in enumerate(ranked_indices):
            cand_id = cand_ids[idx]
            cand_prob = probs[idx]
            feat = cand_features[idx]

            is_s2 = feat[30] == 1.0
            has_addr = feat[14] == 1.0
            c_translit = feat[29] == 1.0

            # Source-specific calibration (Section 20)
            if is_s2 and not has_addr:
                eff_threshold = self.s2_missing_addr_threshold
            elif c_translit:
                eff_threshold = self.s3_translit_threshold
            else:
                eff_threshold = self.base_match_threshold

            # Rule A: Threshold gate
            if cand_prob < eff_threshold:
                break

            # Rule B: Margin from top candidate
            if (best_prob - cand_prob) > eff_margin:
                break

            # Rule C: Slope drop check (if drop from previous candidate > 0.14)
            if rank > 0:
                prev_prob = probs[ranked_indices[rank - 1]]
                if (prev_prob - cand_prob) > 0.14 and rank >= 3:
                    break

            selected.append(cand_id)
            if len(selected) >= effective_max_k:
                break

        return selected
