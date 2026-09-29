"""
ML Challenge 2026 - Business Entity Resolution Pipeline
V3 Multi-Type Hard-Negative Mining Engine (Section 13)

Mines realistic hard negatives across 4 primary collision types:
1. Same name, conflicting address (e.g. different branches / unrelated businesses)
2. Same address, different business name (e.g. multi-tenant commercial buildings)
3. Generic business token collisions ('restaurant', 'hotel', 'store', 'consulting')
4. Transliteration / prefix collisions in multilingual records

Supports customizable negative-to-positive ratios: 1:4, 1:6, 1:8, 1:10.
"""

from collections import defaultdict
import random
import re
import string
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import rapidfuzz.fuzz as fuzz

PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

COMMON_GENERIC = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc",
    "co", "company", "pvt", "private", "the", "and", "of", "in", "for",
    "sa", "sarl", "gmbh", "bv", "services", "enterprises", "solutions",
    "technologies", "group", "holdings", "international", "consulting",
    "restaurant", "hotel", "store", "shop", "trading", "industries",
    "center", "centre", "associates", "management", "system", "systems",
}


def mine_v3_training_dataset(
    s1_records: List[Tuple[str, str, str, str]],
    target_records: List[Tuple[str, str, str, str]],
    ground_truth: Dict[str, Set[str]],
    candidate_generator,
    negative_ratio: int = 6,
    random_state: int = 42,
) -> Tuple[List[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]], np.ndarray, Dict[str, int]]:
    """
    Mines balanced diverse positives and stratified hard negatives.
    Returns: (pairs, labels, negative_type_stats)
    """
    rng = random.Random(random_state)
    target_dict = {t[0]: t for t in target_records}

    positive_pairs = []
    neg_by_type = {
        "same_name_diff_addr": [],
        "same_addr_diff_name": [],
        "generic_collision": [],
        "multi_channel_distractor": [],
    }

    print(f"Mining V3 training dataset from {len(s1_records):,} entities (Target Ratio 1:{negative_ratio})...", flush=True)

    for s1_rec in s1_records:
        s1_id, s1_name, s1_addr, s1_country = s1_rec
        true_targets = ground_truth.get(s1_id, set())

        # 1. Positives
        for tid in true_targets:
            if tid in target_dict:
                positive_pairs.append((s1_rec, target_dict[tid]))

        # 2. Retrieve candidates
        c_idxs = candidate_generator.get_candidates(
            name=s1_name,
            addr=s1_addr,
            country=s1_country,
            limit_per_channel=25,
            max_union=80,
        )

        # 3. Categorize Hard Negatives
        for c_idx in c_idxs:
            cand_rec = target_records[c_idx]
            cand_id = cand_rec[0]
            if cand_id in true_targets:
                continue

            c_name, c_addr, c_country = cand_rec[1], cand_rec[2], cand_rec[3]

            n_sim = fuzz.token_sort_ratio(s1_name, c_name)
            a_sim = fuzz.token_sort_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0

            s1_toks = set(s1_name.translate(PUNCT_TABLE).lower().split())
            c_toks = set(c_name.translate(PUNCT_TABLE).lower().split())
            shared = s1_toks & c_toks

            if n_sim >= 80 and a_sim < 40 and bool(s1_addr and c_addr):
                neg_by_type["same_name_diff_addr"].append((s1_rec, cand_rec))
            elif a_sim >= 80 and n_sim < 50:
                neg_by_type["same_addr_diff_name"].append((s1_rec, cand_rec))
            elif shared and all(t in COMMON_GENERIC for t in shared):
                neg_by_type["generic_collision"].append((s1_rec, cand_rec))
            else:
                neg_by_type["multi_channel_distractor"].append((s1_rec, cand_rec))

    n_pos = len(positive_pairs)
    n_neg_total = int(n_pos * negative_ratio)

    # Allocate across negative types with balanced stratification
    allocations = {
        "same_name_diff_addr": int(n_neg_total * 0.35),
        "same_addr_diff_name": int(n_neg_total * 0.25),
        "generic_collision": int(n_neg_total * 0.20),
        "multi_channel_distractor": int(n_neg_total * 0.20),
    }

    selected_negatives = []
    stats = {}
    for ntype, desired_count in allocations.items():
        pool = neg_by_type[ntype]
        if len(pool) >= desired_count:
            picked = rng.sample(pool, desired_count)
        else:
            picked = pool
        selected_negatives.extend(picked)
        stats[ntype] = len(picked)

    # If still below target, fill from remaining distractors
    remaining_needed = n_neg_total - len(selected_negatives)
    if remaining_needed > 0:
        all_left = []
        for ntype, pool in neg_by_type.items():
            left = [p for p in pool if p not in selected_negatives]
            all_left.extend(left)
        if len(all_left) >= remaining_needed:
            selected_negatives.extend(rng.sample(all_left, remaining_needed))
        else:
            selected_negatives.extend(all_left)

    actual_ratio = len(selected_negatives) / max(n_pos, 1)
    print(f"Mined V3 Dataset: {n_pos:,} diverse positives, {len(selected_negatives):,} stratified hard negatives (Ratio 1:{actual_ratio:.1f}).", flush=True)
    print(f"Negative Stratification: {stats}", flush=True)

    all_pairs = positive_pairs + selected_negatives
    labels = np.array([1] * len(positive_pairs) + [0] * len(selected_negatives), dtype=np.int32)

    indices = list(range(len(all_pairs)))
    rng.shuffle(indices)

    shuffled_pairs = [all_pairs[i] for i in indices]
    shuffled_labels = labels[indices]

    return shuffled_pairs, shuffled_labels, stats
