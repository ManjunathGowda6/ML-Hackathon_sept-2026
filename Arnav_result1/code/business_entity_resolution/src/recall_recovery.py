import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from scripts.multiview import MultiViewRecord

class RecoveryChannel:
    RARE_TOKEN_PAIR = "rare_token_pair"
    CHAR_NGRAM = "char_ngram"
    TRANSLITERATION = "transliteration"
    ADDRESS_ANCHOR = "address_anchor"
    NAME_ADDR_CROSS = "name_addr_cross"
    DOMAIN_HANDLE = "domain_handle"
    TOKEN_ORDER_INVARIANT = "token_order_invariant"
    COMPACT_SIGNATURE = "compact_signature"

class RecallRecoveryRetriever:
    """
    Secondary Recall Recovery Retriever:
    Complementary to Primary MultiChannelRetriever, focusing on:
    - Rare token combinations (Channel A)
    - Character 3-gram signatures (Channel B)
    - Transliteration views (Channel C)
    - Address Anchors: House# + Street, Postal + Street, etc. (Channel D)
    - Cross Name+Address Anchors (Channel E)
    - Domain / URL stem lookups (Channel G)
    - Token Order Invariant (Channel H)
    - Compact Signature (Channel I)
    """
    def __init__(self, max_token_df: int = 15000, max_anchor_df: int = 1500, max_ngram_df: int = 2500):
        self.max_token_df = max_token_df
        self.max_anchor_df = max_anchor_df
        self.max_ngram_df = max_ngram_df
        self.total_docs = 0

        # Country-partitioned indices
        self.country_indices = defaultdict(lambda: {
            'rare_token_pair': defaultdict(list),
            'char_ngram': defaultdict(list),
            'translit_compact': defaultdict(list),
            'translit_token': defaultdict(list),
            'addr_anchor_house_street': defaultdict(list),
            'addr_anchor_postal_street': defaultdict(list),
            'addr_anchor_city_street': defaultdict(list),
            'name_addr_cross': defaultdict(list),
            'domain_stem': defaultdict(list),
            'sorted_stripped_name': defaultdict(list),
            'compact_stripped_name': defaultdict(list),
            'token_df': Counter(),
            'token_addr_df': Counter(),
            'ngram_df': Counter(),
            'doc_count': 0
        })

    def index_target_records(self, target_records: List[MultiViewRecord]):
        self.total_docs = len(target_records)
        print(f"[RecallRecoveryRetriever] Indexing {self.total_docs:,} target records into recovery indices...")

        # Step 1: Frequency counting
        for idx, rec in enumerate(target_records):
            c_idx = self.country_indices[rec.country]
            c_idx['doc_count'] += 1

            for t in set(rec.tokens_name):
                c_idx['token_df'][t] += 1
            for t in set(rec.tokens_translit_name):
                c_idx['token_df'][t] += 1
            for at in set(rec.rare_tokens_address):
                c_idx['token_addr_df'][at] += 1
            for ng in rec.char_3grams_name:
                c_idx['ngram_df'][ng] += 1

        # Step 2: Index population
        for idx, rec in enumerate(target_records):
            c_idx = self.country_indices[rec.country]

            # Channel G: Domain Stem
            if rec.domain_stem and len(rec.domain_stem) >= 3:
                c_idx['domain_stem'][rec.domain_stem].append(idx)

            # Channel C: Transliterated Compact & Transliterated Tokens
            if rec.compact_translit_name and len(rec.compact_translit_name) >= 3:
                c_idx['translit_compact'][rec.compact_translit_name].append(idx)
            for tt in set(rec.tokens_translit_name):
                if 2 <= len(tt) and c_idx['token_df'][tt] <= self.max_token_df:
                    c_idx['translit_token'][tt].append(idx)

            # Channel H: Token Order Invariant (Sorted Stripped Name)
            if rec.legal_stripped_name:
                sort_stripped = " ".join(sorted(rec.legal_stripped_name.split()))
                if len(sort_stripped) >= 3:
                    c_idx['sorted_stripped_name'][sort_stripped].append(idx)

            # Channel I: Compact Stripped Name
            if rec.compact_stripped_name and len(rec.compact_stripped_name) >= 3:
                c_idx['compact_stripped_name'][rec.compact_stripped_name].append(idx)

            # Channel A: Rare Token Combinations (rarest + second rarest name tokens)
            all_name_toks = list(set(rec.tokens_name + rec.tokens_translit_name))
            if len(all_name_toks) >= 2:
                # Sort by document frequency ascending
                sorted_toks = sorted(all_name_toks, key=lambda t: c_idx['token_df'][t])
                t1, t2 = sorted_toks[0], sorted_toks[1]
                if c_idx['token_df'][t1] <= self.max_token_df and c_idx['token_df'][t2] <= self.max_token_df:
                    pair_key = (min(t1, t2), max(t1, t2))
                    c_idx['rare_token_pair'][pair_key].append(idx)

            # Channel D: Address Anchors
            # Anchor 1: House Number + Street Rare Token
            if rec.house_number and rec.rare_tokens_address:
                for at in rec.rare_tokens_address[:2]:
                    if c_idx['token_addr_df'][at] <= self.max_anchor_df:
                        c_idx['addr_anchor_house_street'][(rec.house_number, at)].append(idx)

            # Anchor 2: Postal Code + Street Rare Token
            if rec.postal_code and rec.rare_tokens_address:
                for at in rec.rare_tokens_address[:2]:
                    if c_idx['token_addr_df'][at] <= self.max_anchor_df:
                        c_idx['addr_anchor_postal_street'][(rec.postal_code, at)].append(idx)

            # Anchor 3: Postal Code + House Number
            if rec.postal_code and rec.house_number:
                c_idx['addr_anchor_postal_street'][(rec.postal_code, rec.house_number)].append(idx)

            # Channel E: Cross Name+Address (Rare name token + Street token)
            if all_name_toks and rec.rare_tokens_address:
                # Rarest name token + rarest address token
                r_name = min(all_name_toks, key=lambda t: c_idx['token_df'][t])
                r_addr = min(rec.rare_tokens_address, key=lambda t: c_idx['token_addr_df'][t])
                if c_idx['token_df'][r_name] <= self.max_token_df and c_idx['token_addr_df'][r_addr] <= self.max_anchor_df:
                    c_idx['name_addr_cross'][(r_name, r_addr)].append(idx)

            # Channel B: Infrequent Character 3-grams
            if rec.char_3grams_name and len(rec.compact_stripped_name or rec.compact_name) <= 25:
                rare_3grams = [ng for ng in rec.char_3grams_name if c_idx['ngram_df'][ng] <= self.max_ngram_df]
                for ng in rare_3grams[:4]:
                    c_idx['char_ngram'][ng].append(idx)

        print("[RecallRecoveryRetriever] Indexing complete.")

    def retrieve_recovery_candidates(self, query_rec: MultiViewRecord, max_per_channel: int = 25, recovery_budget: int = 25) -> Dict[int, dict]:
        """
        Retrieves secondary recovery candidates for an S1 entity.
        """
        recovery_cands = defaultdict(lambda: {
            'channels': set(),
            'channel_ranks': {},
            'channel_scores': {},
            'is_recovery': 1,
            'recovery_priority': 0.0
        })

        target_indices = []
        if query_rec.country and query_rec.country in self.country_indices:
            target_indices.append(self.country_indices[query_rec.country])
        else:
            target_indices = list(self.country_indices.values())

        for c_idx in target_indices:
            # 1. Domain Stem Lookup (Channel G)
            # If query raw_name is or contains domain stem, or matches target domain
            q_stem = query_rec.domain_stem or query_rec.compact_stripped_name
            if q_stem and len(q_stem) >= 3 and q_stem in c_idx['domain_stem']:
                hits = c_idx['domain_stem'][q_stem][:max_per_channel]
                for rank, tidx in enumerate(hits, 1):
                    recovery_cands[tidx]['channels'].add(RecoveryChannel.DOMAIN_HANDLE)
                    recovery_cands[tidx]['channel_ranks'][RecoveryChannel.DOMAIN_HANDLE] = rank
                    recovery_cands[tidx]['channel_scores'][RecoveryChannel.DOMAIN_HANDLE] = 1.0
                    recovery_cands[tidx]['recovery_priority'] += 5.0

            # 2. Transliteration View (Channel C)
            # Query transliterated compact or transliterated tokens against target
            q_trans_compact = query_rec.compact_translit_name or query_rec.compact_name
            if q_trans_compact and len(q_trans_compact) >= 3 and q_trans_compact in c_idx['translit_compact']:
                hits = c_idx['translit_compact'][q_trans_compact][:max_per_channel]
                for rank, tidx in enumerate(hits, 1):
                    recovery_cands[tidx]['channels'].add(RecoveryChannel.TRANSLITERATION)
                    recovery_cands[tidx]['channel_ranks'][RecoveryChannel.TRANSLITERATION] = rank
                    recovery_cands[tidx]['channel_scores'][RecoveryChannel.TRANSLITERATION] = 0.95
                    recovery_cands[tidx]['recovery_priority'] += 4.5

            # Transliterated tokens
            all_q_trans_toks = list(set(query_rec.tokens_translit_name + query_rec.tokens_name))
            if all_q_trans_toks:
                sorted_trans_toks = sorted(all_q_trans_toks, key=lambda t: c_idx['token_df'][t])
                for tt in sorted_trans_toks[:2]:
                    if tt in c_idx['translit_token']:
                        for rank, tidx in enumerate(c_idx['translit_token'][tt][:max_per_channel], 1):
                            recovery_cands[tidx]['channels'].add(RecoveryChannel.TRANSLITERATION)
                            recovery_cands[tidx]['channel_ranks'][RecoveryChannel.TRANSLITERATION] = rank
                            recovery_cands[tidx]['channel_scores'][RecoveryChannel.TRANSLITERATION] = 0.85
                            recovery_cands[tidx]['recovery_priority'] += 3.0

            # 3. Token Order Invariant (Channel H)
            if query_rec.legal_stripped_name:
                q_sort_stripped = " ".join(sorted(query_rec.legal_stripped_name.split()))
                if len(q_sort_stripped) >= 3 and q_sort_stripped in c_idx['sorted_stripped_name']:
                    hits = c_idx['sorted_stripped_name'][q_sort_stripped][:max_per_channel]
                    for rank, tidx in enumerate(hits, 1):
                        recovery_cands[tidx]['channels'].add(RecoveryChannel.TOKEN_ORDER_INVARIANT)
                        recovery_cands[tidx]['channel_ranks'][RecoveryChannel.TOKEN_ORDER_INVARIANT] = rank
                        recovery_cands[tidx]['channel_scores'][RecoveryChannel.TOKEN_ORDER_INVARIANT] = 0.95
                        recovery_cands[tidx]['recovery_priority'] += 4.5

            # 4. Compact Signature (Channel I)
            if query_rec.compact_stripped_name and query_rec.compact_stripped_name in c_idx['compact_stripped_name']:
                hits = c_idx['compact_stripped_name'][query_rec.compact_stripped_name][:max_per_channel]
                for rank, tidx in enumerate(hits, 1):
                    recovery_cands[tidx]['channels'].add(RecoveryChannel.COMPACT_SIGNATURE)
                    recovery_cands[tidx]['channel_ranks'][RecoveryChannel.COMPACT_SIGNATURE] = rank
                    recovery_cands[tidx]['channel_scores'][RecoveryChannel.COMPACT_SIGNATURE] = 0.95
                    recovery_cands[tidx]['recovery_priority'] += 4.5

            # 5. Rare Token Combinations (Channel A)
            if len(all_q_trans_toks) >= 2:
                sorted_toks = sorted(all_q_trans_toks, key=lambda t: c_idx['token_df'][t])
                t1, t2 = sorted_toks[0], sorted_toks[1]
                pair_key = (min(t1, t2), max(t1, t2))
                if pair_key in c_idx['rare_token_pair']:
                    hits = c_idx['rare_token_pair'][pair_key][:max_per_channel]
                    for rank, tidx in enumerate(hits, 1):
                        recovery_cands[tidx]['channels'].add(RecoveryChannel.RARE_TOKEN_PAIR)
                        recovery_cands[tidx]['channel_ranks'][RecoveryChannel.RARE_TOKEN_PAIR] = rank
                        recovery_cands[tidx]['channel_scores'][RecoveryChannel.RARE_TOKEN_PAIR] = 0.90
                        recovery_cands[tidx]['recovery_priority'] += 4.0

            # 6. Address Anchors (Channel D)
            # House + Street
            if query_rec.house_number and query_rec.rare_tokens_address:
                for at in query_rec.rare_tokens_address[:2]:
                    key = (query_rec.house_number, at)
                    if key in c_idx['addr_anchor_house_street']:
                        hits = c_idx['addr_anchor_house_street'][key][:max_per_channel]
                        for rank, tidx in enumerate(hits, 1):
                            recovery_cands[tidx]['channels'].add(RecoveryChannel.ADDRESS_ANCHOR)
                            recovery_cands[tidx]['channel_ranks'][RecoveryChannel.ADDRESS_ANCHOR] = rank
                            recovery_cands[tidx]['channel_scores'][RecoveryChannel.ADDRESS_ANCHOR] = 0.85
                            recovery_cands[tidx]['recovery_priority'] += 3.5

            # Postal + Street or Postal + House#
            if query_rec.postal_code:
                if query_rec.house_number:
                    key = (query_rec.postal_code, query_rec.house_number)
                    if key in c_idx['addr_anchor_postal_street']:
                        hits = c_idx['addr_anchor_postal_street'][key][:max_per_channel]
                        for rank, tidx in enumerate(hits, 1):
                            recovery_cands[tidx]['channels'].add(RecoveryChannel.ADDRESS_ANCHOR)
                            recovery_cands[tidx]['channel_ranks'][RecoveryChannel.ADDRESS_ANCHOR] = rank
                            recovery_cands[tidx]['channel_scores'][RecoveryChannel.ADDRESS_ANCHOR] = 0.85
                            recovery_cands[tidx]['recovery_priority'] += 3.5

                if query_rec.rare_tokens_address:
                    for at in query_rec.rare_tokens_address[:2]:
                        key = (query_rec.postal_code, at)
                        if key in c_idx['addr_anchor_postal_street']:
                            hits = c_idx['addr_anchor_postal_street'][key][:max_per_channel]
                            for rank, tidx in enumerate(hits, 1):
                                recovery_cands[tidx]['channels'].add(RecoveryChannel.ADDRESS_ANCHOR)
                                recovery_cands[tidx]['channel_ranks'][RecoveryChannel.ADDRESS_ANCHOR] = rank
                                recovery_cands[tidx]['channel_scores'][RecoveryChannel.ADDRESS_ANCHOR] = 0.80
                                recovery_cands[tidx]['recovery_priority'] += 3.0

            # 7. Name + Address Cross (Channel E)
            if all_q_trans_toks and query_rec.rare_tokens_address:
                r_name = min(all_q_trans_toks, key=lambda t: c_idx['token_df'][t])
                r_addr = min(query_rec.rare_tokens_address, key=lambda t: c_idx['token_addr_df'][t])
                key = (r_name, r_addr)
                if key in c_idx['name_addr_cross']:
                    hits = c_idx['name_addr_cross'][key][:max_per_channel]
                    for rank, tidx in enumerate(hits, 1):
                        recovery_cands[tidx]['channels'].add(RecoveryChannel.NAME_ADDR_CROSS)
                        recovery_cands[tidx]['channel_ranks'][RecoveryChannel.NAME_ADDR_CROSS] = rank
                        recovery_cands[tidx]['channel_scores'][RecoveryChannel.NAME_ADDR_CROSS] = 0.90
                        recovery_cands[tidx]['recovery_priority'] += 4.0

            # 8. Character 3-grams (Channel B)
            if query_rec.char_3grams_name and len(query_rec.compact_stripped_name or query_rec.compact_name) <= 25:
                ngram_hits = Counter()
                for ng in query_rec.char_3grams_name:
                    if ng in c_idx['char_ngram']:
                        for tidx in c_idx['char_ngram'][ng][:max_per_channel]:
                            ngram_hits[tidx] += 1
                
                # Require at least 2 shared rare n-grams
                for tidx, hit_count in ngram_hits.most_common(max_per_channel):
                    if hit_count >= 2:
                        recovery_cands[tidx]['channels'].add(RecoveryChannel.CHAR_NGRAM)
                        recovery_cands[tidx]['channel_ranks'][RecoveryChannel.CHAR_NGRAM] = 1
                        recovery_cands[tidx]['channel_scores'][RecoveryChannel.CHAR_NGRAM] = 0.75
                        recovery_cands[tidx]['recovery_priority'] += 2.0

        # Controlled candidate union: sort by recovery_priority and prune to recovery_budget
        if len(recovery_cands) > recovery_budget:
            sorted_items = sorted(recovery_cands.items(), key=lambda it: it[1]['recovery_priority'], reverse=True)
            recovery_cands = dict(sorted_items[:recovery_budget])

        return recovery_cands


class CandidateEvidenceMerger:
    """
    Unites Primary Candidates and Secondary Recovery Candidates into a controlled,
    evidence-tagged candidate pool with candidate categories (A, B, C, D, E).
    """
    @staticmethod
    def merge(primary_cands: Dict[int, dict], recovery_cands: Dict[int, dict], max_total_budget: int = 100) -> Dict[int, dict]:
        merged = {}

        # 1. Process primary candidates
        for tidx, p_info in primary_cands.items():
            merged[tidx] = {
                'channels': set(p_info['channels']),
                'channel_ranks': dict(p_info['channel_ranks']),
                'channel_scores': dict(p_info['channel_scores']),
                'token_hits': p_info.get('token_hits', 0),
                'max_idf': p_info.get('max_idf', 0.0),
                'sum_idf': p_info.get('sum_idf', 0.0),
                'is_primary': 1,
                'is_recovery': 0,
            }

        # 2. Enrich or add recovery candidates
        for tidx, r_info in recovery_cands.items():
            if tidx in merged:
                # Candidate found by both!
                m = merged[tidx]
                m['channels'].update(r_info['channels'])
                m['channel_ranks'].update(r_info['channel_ranks'])
                m['channel_scores'].update(r_info['channel_scores'])
                m['is_recovery'] = 1
            else:
                # Recovery-only candidate
                merged[tidx] = {
                    'channels': set(r_info['channels']),
                    'channel_ranks': dict(r_info['channel_ranks']),
                    'channel_scores': dict(r_info['channel_scores']),
                    'token_hits': 0,
                    'max_idf': 0.0,
                    'sum_idf': 0.0,
                    'is_primary': 0,
                    'is_recovery': 1,
                }

        # 3. Categorize candidates (Section 16 of specification)
        for tidx, data in merged.items():
            channels = data['channels']
            n_ch = len(channels)
            ranks = list(data['channel_ranks'].values())
            data['channel_count'] = n_ch
            data['best_retrieval_rank'] = min(ranks) if ranks else 99
            scores = list(data['channel_scores'].values())
            data['best_retrieval_score'] = max(scores) if scores else 0.0

            # Exact strong channels
            has_strong_exact = bool(
                {'exact_name', 'exact_stripped', 'exact_compact', 'compact_signature', 
                 'domain_handle', 'token_order_invariant'}.intersection(channels)
            )

            if has_strong_exact:
                data['candidate_category'] = 'D'  # Exact / near-exact strong evidence
            elif n_ch >= 3:
                data['candidate_category'] = 'C'  # 3+ independent channels
            elif n_ch == 2:
                data['candidate_category'] = 'B'  # 2 independent channels
            elif data['is_recovery'] == 1 and data['is_primary'] == 0:
                data['candidate_category'] = 'E'  # Recovery-only candidate
            else:
                data['candidate_category'] = 'A'  # 1 weak retrieval channel

        # 4. Budget enforcement
        if len(merged) > max_total_budget:
            def merge_priority(item):
                tidx, d = item
                cat_rank = {'D': 5, 'C': 4, 'B': 3, 'E': 2, 'A': 1}.get(d['candidate_category'], 0)
                return (cat_rank, d['channel_count'], d['best_retrieval_score'], -d['best_retrieval_rank'])

            sorted_items = sorted(merged.items(), key=merge_priority, reverse=True)
            merged = dict(sorted_items[:max_total_budget])

        return merged
