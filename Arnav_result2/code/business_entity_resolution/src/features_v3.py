"""
ML Challenge 2026 - Business Entity Resolution Pipeline
V3 Discriminative Feature Engineering (Sections 14 & 15)

Computes 32 discriminative features per candidate pair:
1. Name similarity signals (ratio, token sort, token set, suffix-stripped, slug, sorted slug, 3-gram, acronym)
2. Learned corpus frequency & IDF features (devalues generic words: 'hotel', 'restaurant', 'store', 'services')
3. Address component signals (token sort, ratio, house number agreement/conflict, postal code agreement/conflict, state conflict)
4. Address joint key match (house_num + street_word)
5. Cross-field interaction signals (name strong + address conflict, address strong + name conflict)
6. Retrieval evidence signals (number of retrieval channels, best channel rank)
7. Source & Multilingual signals (transliteration candidate, is_source2, is_source3, country agreement)
"""

from collections import Counter
import math
import re
import string
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import rapidfuzz.fuzz as fuzz

PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

SUFFIXES_PATTERN = re.compile(
    r"\b(private limited|pvt ltd|pvt limited|private ltd|incorporated|corporation|enterprise|limited|pvt|ltd|llc|inc|corp|co|llp|gmbh|sarl|sa)\b",
    re.IGNORECASE,
)

GENERIC_STOPWORDS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc",
    "co", "company", "pvt", "private", "the", "and", "of", "in", "for",
    "sa", "sarl", "gmbh", "bv", "services", "enterprises", "solutions",
    "technologies", "group", "holdings", "international", "consulting",
    "restaurant", "hotel", "store", "shop", "trading", "industries",
    "center", "centre", "associates", "management", "system", "systems",
    "india", "usa", "us", "france", "north", "south", "east", "west",
    "market", "medical", "school", "cafe",
}

KNOWN_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "delhi", "maharashtra", "karnataka", "tamil nadu", "gujarat",
    "haryana", "punjab", "rajasthan", "uttar pradesh", "west bengal", "kerala", "bihar",
}

FEATURE_COLUMNS_V3 = [
    "name_ratio",                   # 0
    "name_token_sort",              # 1
    "name_token_set",               # 2
    "name_suffix_stripped_sim",     # 3
    "name_exact_norm",              # 4
    "name_slug_exact",              # 5
    "sorted_slug_match",            # 6
    "name_len_diff",                # 7
    "char_3gram_jaccard",           # 8
    "acronym_match",                # 9
    "name_idf_overlap",             # 10
    "name_idf_jaccard",             # 11
    "generic_only_match",           # 12
    "rare_token_count",             # 13
    "both_have_address",            # 14
    "addr_token_sort",              # 15
    "addr_ratio",                   # 16
    "addr_joint_match",             # 17
    "same_house_number",            # 18
    "conflicting_house_number",     # 19
    "same_postal_code",             # 20
    "conflicting_postal_code",      # 21
    "state_conflict",               # 22
    "name_addr_joint",              # 23
    "high_name_missing_addr",       # 24
    "name_high_addr_conflict",      # 25
    "addr_high_name_conflict",      # 26
    "retrieval_num_channels",       # 27
    "retrieval_best_rank",          # 28
    "transliteration_candidate",    # 29
    "is_source2",                   # 30
    "is_source3",                   # 31
]


def clean_tokens(text: str) -> List[str]:
    s = text.translate(PUNCT_TABLE).lower()
    return [t for t in s.split() if len(t) >= 2]


def make_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def make_sorted_token_slug(text: str) -> str:
    toks = sorted([t for t in text.translate(PUNCT_TABLE).lower().split() if len(t) >= 3 and t not in GENERIC_STOPWORDS])
    return "".join(toks) if toks else make_slug(text)


class V3FeatureExtractor:
    """
    32-Feature Extraction Engine with Corpus Frequency IDF and Channel Evidence Signals.
    """

    def __init__(self, target_records: List[Tuple[str, str, str, str]]):
        self.num_targets = len(target_records)
        print(f"Initializing V3 Feature Extractor across {self.num_targets:,} target records...", flush=True)

        self.token_df = Counter()
        for _, name, _, _ in target_records:
            toks = set(clean_tokens(name))
            for t in toks:
                self.token_df[t] += 1

        self.default_idf = math.log(1.0 + self.num_targets / 2.0)

    def get_idf(self, token: str) -> float:
        df = self.token_df.get(token, 0)
        if df == 0:
            return self.default_idf
        return math.log(1.0 + self.num_targets / (df + 1.0))

    def extract_pair_features(
        self,
        s1_rec: Tuple[str, str, str, str],
        target_rec: Tuple[str, str, str, str],
        evidence_meta: Optional[Tuple[Set[str], int]] = None,
    ) -> List[float]:
        _, n1, a1, c1 = s1_rec
        tid, n2, a2, c2 = target_rec

        n1_clean = n1.translate(PUNCT_TABLE).lower().strip()
        n2_clean = n2.translate(PUNCT_TABLE).lower().strip()

        # 1. Name similarities
        name_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0 if (n1_clean and n2_clean) else 0.0
        name_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0 if (n1_clean and n2_clean) else 0.0
        name_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0 if (n1_clean and n2_clean) else 0.0

        n1_ns = SUFFIXES_PATTERN.sub("", n1_clean).strip()
        n2_ns = SUFFIXES_PATTERN.sub("", n2_clean).strip()
        name_ns_sim = fuzz.token_sort_ratio(n1_ns, n2_ns) / 100.0 if (n1_ns and n2_ns) else name_sort
        name_exact_norm = 1.0 if (n1_ns and n1_ns == n2_ns) else 0.0

        slug1 = make_slug(n1_clean)
        slug2 = make_slug(n2_clean)
        name_slug_exact = 1.0 if (slug1 and slug1 == slug2) else 0.0

        sorted_slug1 = make_sorted_token_slug(n1_clean)
        sorted_slug2 = make_sorted_token_slug(n2_clean)
        sorted_slug_match = 1.0 if (sorted_slug1 and sorted_slug1 == sorted_slug2) else 0.0

        l1, l2 = len(n1_clean), len(n2_clean)
        name_len_diff = abs(l1 - l2) / max(l1, l2, 1)

        g1 = {slug1[i : i + 3] for i in range(len(slug1) - 2)} if len(slug1) >= 3 else ({slug1} if slug1 else set())
        g2 = {slug2[i : i + 3] for i in range(len(slug2) - 2)} if len(slug2) >= 3 else ({slug2} if slug2 else set())
        char_3gram_jaccard = len(g1 & g2) / len(g1 | g2) if (g1 and g2) else 0.0

        toks1 = [t for t in n1_clean.split() if t not in GENERIC_STOPWORDS]
        toks2 = [t for t in n2_clean.split() if t not in GENERIC_STOPWORDS]
        acr1 = "".join(t[0] for t in toks1 if t)
        acr2 = "".join(t[0] for t in toks2 if t)
        acronym_match = 1.0 if (len(acr1) >= 2 and len(acr2) >= 2 and (acr1 == slug2 or acr2 == slug1 or acr1 == acr2)) else 0.0

        # 2. Corpus Frequency & IDF
        all_toks1 = set(n1_clean.split())
        all_toks2 = set(n2_clean.split())
        shared_toks = all_toks1 & all_toks2
        union_toks = all_toks1 | all_toks2

        idf_overlap = sum(self.get_idf(t) for t in shared_toks)
        total_idf = sum(self.get_idf(t) for t in union_toks) if union_toks else 1.0
        idf_jaccard = idf_overlap / total_idf if total_idf > 0 else 0.0

        generic_only = 1.0 if (shared_toks and all(t in GENERIC_STOPWORDS for t in shared_toks)) else 0.0
        rare_token_count = float(sum(1 for t in shared_toks if self.get_idf(t) > 7.0 and t not in GENERIC_STOPWORDS))

        # 3. Address Features
        both_have_addr = 1.0 if (a1.strip() and a2.strip()) else 0.0
        addr_clean1 = a1.translate(PUNCT_TABLE).lower().strip()
        addr_clean2 = a2.translate(PUNCT_TABLE).lower().strip()

        addr_sort = fuzz.token_sort_ratio(addr_clean1, addr_clean2) / 100.0 if both_have_addr else 0.0
        addr_ratio = fuzz.ratio(addr_clean1, addr_clean2) / 100.0 if both_have_addr else 0.0

        nums1 = set(re.findall(r"\b\d+\b", a1))
        nums2 = set(re.findall(r"\b\d+\b", a2))
        house1 = {n for n in nums1 if 1 <= len(n) <= 4}
        house2 = {n for n in nums2 if 1 <= len(n) <= 4}

        same_house = 1.0 if (house1 and house2 and (house1 & house2)) else 0.0
        conflict_house = 1.0 if (house1 and house2 and not (house1 & house2)) else 0.0

        # Address Joint Key Agreement (house_num + street_word)
        addr_toks1 = {w for w in clean_tokens(addr_clean1) if len(w) >= 4 and not w.isdigit()}
        addr_toks2 = {w for w in clean_tokens(addr_clean2) if len(w) >= 4 and not w.isdigit()}
        addr_joint_match = 1.0 if (same_house and (addr_toks1 & addr_toks2)) else 0.0

        post1 = {n for n in nums1 if len(n) in (5, 6)}
        post2 = {n for n in nums2 if len(n) in (5, 6)}
        same_postal = 1.0 if (post1 and post2 and (post1 & post2)) else 0.0
        conflict_postal = 1.0 if (post1 and post2 and not (post1 & post2)) else 0.0

        state_conflict = 0.0
        if both_have_addr and addr_sort < 0.35:
            st1 = set(addr_clean1.split()) & KNOWN_STATES
            st2 = set(addr_clean2.split()) & KNOWN_STATES
            if st1 and st2 and not (st1 & st2):
                state_conflict = 1.0

        # 4. Cross-Field Interactions
        name_addr_joint = (name_sort + addr_sort) / 2.0 if both_have_addr else name_sort
        high_name_missing_addr = 1.0 if (name_sort >= 0.85 and both_have_addr == 0.0) else 0.0
        name_high_addr_conflict = 1.0 if (name_sort >= 0.80 and both_have_addr and (conflict_house or conflict_postal or state_conflict)) else 0.0
        addr_high_name_conflict = 1.0 if (both_have_addr and addr_sort >= 0.80 and name_sort < 0.45) else 0.0

        # 5. Retrieval Evidence (Channels & Rank)
        if evidence_meta:
            channels, rank = evidence_meta
            num_channels = float(len(channels))
            best_rank = float(min(rank, 30)) / 30.0
        else:
            num_channels = 1.0
            best_rank = 0.5

        # 6. Multilingual & Source Signals
        is_translit = 0.0
        if not n1.isascii() or not n2.isascii():
            if both_have_addr and (addr_joint_match or addr_sort >= 0.70):
                is_translit = 1.0
            elif char_3gram_jaccard >= 0.40:
                is_translit = 1.0

        is_s2 = 1.0 if tid.startswith("S2") else 0.0
        is_s3 = 1.0 if tid.startswith("S3") else 0.0

        return [
            name_ratio,
            name_sort,
            name_set,
            name_ns_sim,
            name_exact_norm,
            name_slug_exact,
            sorted_slug_match,
            name_len_diff,
            char_3gram_jaccard,
            acronym_match,
            idf_overlap,
            idf_jaccard,
            generic_only,
            rare_token_count,
            both_have_addr,
            addr_sort,
            addr_ratio,
            addr_joint_match,
            same_house,
            conflict_house,
            same_postal,
            conflict_postal,
            state_conflict,
            name_addr_joint,
            high_name_missing_addr,
            name_high_addr_conflict,
            addr_high_name_conflict,
            num_channels,
            best_rank,
            is_translit,
            is_s2,
            is_s3,
        ]

    def extract_batch_features(
        self,
        pairs: List[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]],
        evidence_metas: Optional[List[Tuple[Set[str], int]]] = None,
    ) -> np.ndarray:
        if evidence_metas:
            feats = [self.extract_pair_features(s1, tgt, meta) for (s1, tgt), meta in zip(pairs, evidence_metas)]
        else:
            feats = [self.extract_pair_features(s1, tgt) for s1, tgt in pairs]
        return np.array(feats, dtype=np.float32)
