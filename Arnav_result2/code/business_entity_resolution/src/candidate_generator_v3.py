"""
ML Challenge 2026 - Business Entity Resolution Pipeline
V3 Candidate Generation Engine with Evidence Tracking & Channel-Aware Budgets

Key Advancements (Sections 11 & 12):
1. Multi-Channel Evidence Tracking: Tracks retrieval channel set, best rank, and channel diversity
2. Channel-Aware Budgets: High-precision channels prioritized; generic token flooding strictly budgeted
3. Address Joint Keys: (house_num + street_word) provides near-unique blocking in 10M records
4. Word-Order-Invariant Sorted Slugs: Resolves 'Hotel Royal' vs 'Royal Hotel'
5. Multi-Channel Support Filter: Prioritizes candidates supported by >= 2 independent channels
6. Recovery Layer: Activates when candidate pool has sparse evidence (< 5 candidates)
"""

from collections import defaultdict, Counter
import re
import string
import time
import unicodedata
from typing import Dict, List, Set, Tuple

PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

COMMON_GENERIC_TOKENS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc",
    "co", "company", "pvt", "private", "the", "and", "of", "in", "for",
    "sa", "sarl", "gmbh", "bv", "services", "enterprises", "solutions",
    "technologies", "group", "holdings", "international", "consulting",
    "restaurant", "hotel", "store", "shop", "trading", "industries",
    "center", "centre", "associates", "management", "system", "systems",
    "india", "usa", "us", "france",
}


def clean_tokens(text: str) -> List[str]:
    s = text.translate(PUNCT_TABLE).lower()
    if not s.isascii():
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    return [t for t in s.split() if len(t) >= 3 and t not in COMMON_GENERIC_TOKENS]


def make_slug(text: str) -> str:
    s = text.lower()
    if not s.isascii():
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r"[^a-z0-9]", "", s)


def make_sorted_token_slug(text: str) -> str:
    toks = sorted(clean_tokens(text))
    return "".join(toks) if toks else make_slug(text)


def extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+\b", text))


class V3EvidenceCandidateGenerator:
    """
    V3 Multi-Channel Candidate Generator with Channel Evidence Tracking.
    """

    def __init__(self, target_records: List[Tuple[str, str, str, str]], max_token_freq: int = 4000):
        self.records = target_records
        t0 = time.time()
        print(f"Building V3 Candidate Index across {len(target_records):,} records...", flush=True)

        self.by_country = defaultdict(lambda: {
            "exact_name": defaultdict(list),
            "slug": defaultdict(list),
            "sorted_slug": defaultdict(list),
            "slug_prefix": defaultdict(list),
            "addr_joint": defaultdict(list),
            "postal_code": defaultdict(list),
            "addr_num": defaultdict(list),
            "name_tokens": defaultdict(list),
        })

        for idx, (tid, name, addr, country) in enumerate(target_records):
            c = country.upper() if country else "UNKNOWN"
            c_dict = self.by_country[c]

            # 1. Exact normalized name
            clean_name = " ".join(clean_tokens(name))
            if clean_name:
                c_dict["exact_name"][clean_name].append(idx)

            # 2. Slugs (regular + sorted-token)
            slug = make_slug(name)
            if slug:
                c_dict["slug"][slug].append(idx)
                if len(slug) >= 5:
                    c_dict["slug_prefix"][slug[:5]].append(idx)

            sorted_slug = make_sorted_token_slug(name)
            if sorted_slug and sorted_slug != slug:
                c_dict["sorted_slug"][sorted_slug].append(idx)

            # 3. Informative name tokens
            toks = clean_tokens(name)
            for tok in toks:
                c_dict["name_tokens"][tok].append(idx)

            # 4. Address numbers, postals, and joint keys
            nums = extract_numbers(addr)
            addr_words = [w for w in clean_tokens(addr) if len(w) >= 4 and not w.isdigit()]

            for n in nums:
                if len(n) <= 6:
                    c_dict["addr_num"][n].append(idx)
                    for w in addr_words[:2]:
                        joint_key = f"{n}_{w}"
                        c_dict["addr_joint"][joint_key].append(idx)

                if len(n) in (5, 6):
                    c_dict["postal_code"][n].append(idx)

        # Upper bound pruning for heavily saturated posting lists
        pruned_count = 0
        for c, c_dict in self.by_country.items():
            for tok, lst in list(c_dict["name_tokens"].items()):
                if len(lst) > max_token_freq:
                    c_dict["name_tokens"][tok] = lst[:max_token_freq]
                    pruned_count += 1

        print(f"V3 Candidate Index operational in {time.time() - t0:.2f} s. Pruned {pruned_count} lists.", flush=True)

    def get_candidates(
        self,
        name: str,
        addr: str,
        country: str,
        limit_per_channel: int = 30,
        max_union: int = 80,
    ) -> List[int]:
        """Returns standard list of candidate target indices (up to max_union)."""
        cands_with_meta = self.get_candidates_with_evidence(
            name, addr, country, limit_per_channel, max_union
        )
        return [c[0] for c in cands_with_meta]

    def get_candidates_with_evidence(
        self,
        name: str,
        addr: str,
        country: str,
        limit_per_channel: int = 30,
        max_union: int = 80,
    ) -> List[Tuple[int, Set[str], int]]:
        """
        Returns candidates with evidence metadata:
        List of (target_idx, channel_set, best_rank)
        """
        c = country.upper() if country else "UNKNOWN"
        c_dict = self.by_country.get(c, None)
        if not c_dict:
            c_dict = self.by_country.get("UNKNOWN", None)
            if not c_dict:
                return []

        cand_evidence = defaultdict(lambda: {"channels": set(), "best_rank": 999})

        def record_matches(matches, channel_name, cap):
            for rank, midx in enumerate(matches[:cap]):
                entry = cand_evidence[midx]
                entry["channels"].add(channel_name)
                if rank < entry["best_rank"]:
                    entry["best_rank"] = rank

        # Channel 1: Exact Normalized Name
        clean_name = " ".join(clean_tokens(name))
        if clean_name in c_dict["exact_name"]:
            record_matches(c_dict["exact_name"][clean_name], "exact_name", limit_per_channel)

        # Channel 2: Exact Slugs (Standard & Word-Order-Sorted)
        slug = make_slug(name)
        if slug in c_dict["slug"]:
            record_matches(c_dict["slug"][slug], "slug", limit_per_channel)

        sorted_slug = make_sorted_token_slug(name)
        if sorted_slug in c_dict["sorted_slug"]:
            record_matches(c_dict["sorted_slug"][sorted_slug], "sorted_slug", limit_per_channel)

        # Channel 3: 5-char Slug Prefix
        if len(slug) >= 5:
            sp = slug[:5]
            if sp in c_dict["slug_prefix"]:
                matches = c_dict["slug_prefix"][sp]
                if len(matches) <= 600:
                    record_matches(matches, "slug_prefix", limit_per_channel)

        # Channel 4: Distinctive Name Tokens (Sorted by Rarity)
        toks = clean_tokens(name)
        if toks:
            toks_with_freq = [(tok, len(c_dict["name_tokens"].get(tok, []))) for tok in toks if tok in c_dict["name_tokens"]]
            toks_with_freq.sort(key=lambda x: x[1])

            for tok, freq in toks_with_freq[:3]:
                if freq <= 1500:
                    record_matches(c_dict["name_tokens"][tok], "name_token", limit_per_channel)

        # Channel 5: Address Joint Key (house_num + street_word)
        nums = extract_numbers(addr)
        addr_words = [w for w in clean_tokens(addr) if len(w) >= 4 and not w.isdigit()]
        for n in nums:
            if len(n) <= 6:
                for w in addr_words[:2]:
                    jk = f"{n}_{w}"
                    if jk in c_dict["addr_joint"]:
                        matches = c_dict["addr_joint"][jk]
                        if len(matches) <= 300:
                            record_matches(matches, "addr_joint", limit_per_channel)

            if len(n) in (5, 6) and n in c_dict["postal_code"]:
                matches = c_dict["postal_code"][n]
                if len(matches) <= 300:
                    record_matches(matches, "postal_code", 15)

        # Channel 6: Recovery Layer (Activates when evidence is sparse, < 5 candidates)
        if len(cand_evidence) < 5:
            if len(slug) >= 4:
                sp4 = slug[:4]
                if sp4 in c_dict["slug_prefix"]:
                    matches = c_dict["slug_prefix"][sp4]
                    if len(matches) <= 200:
                        record_matches(matches, "recovery_slug4", 10)

            addr_toks = set(clean_tokens(addr))
            for n in nums:
                if n in c_dict["addr_num"]:
                    matches = c_dict["addr_num"][n]
                    if len(matches) <= 350 and addr_toks:
                        for midx in matches[:40]:
                            t_addr_toks = set(clean_tokens(self.records[midx][2]))
                            if len(addr_toks & t_addr_toks) >= 1:
                                entry = cand_evidence[midx]
                                entry["channels"].add("recovery_addr_num")
                                if 15 < entry["best_rank"]:
                                    entry["best_rank"] = 15

        # Quality Control Ranking:
        # Score candidates by:
        # 1. Number of channels (multi-channel supported candidates ranked first)
        # 2. Priority of channel (exact_name > slug > addr_joint > slug_prefix > name_token)
        # 3. Best channel rank
        def candidate_priority(item):
            midx, data = item
            chans = data["channels"]
            n_chans = len(chans)
            high_pri = ("exact_name" in chans or "slug" in chans or "sorted_slug" in chans or "addr_joint" in chans)
            return (n_chans >= 2, high_pri, n_chans, -data["best_rank"])

        sorted_cands = sorted(cand_evidence.items(), key=candidate_priority, reverse=True)

        results = []
        for midx, data in sorted_cands[:max_union]:
            results.append((midx, data["channels"], data["best_rank"]))

        return results
