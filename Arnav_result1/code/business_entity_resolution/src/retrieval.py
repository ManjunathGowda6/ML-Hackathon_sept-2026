import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

class RetrievalChannel:
    EXACT_NAME = "exact_name"
    EXACT_STRIPPED = "exact_stripped"
    EXACT_COMPACT = "exact_compact"
    RARE_NAME_TOKEN = "rare_name_token"
    POSTAL_PREFIX = "postal_prefix"
    ADDR_RARE_TOKEN = "addr_rare_token"
    NAME_ADDR_COMBINED = "name_addr_combined"

class MultiChannelRetriever:
    def __init__(self, max_token_df: int = 18000, max_addr_df: int = 10000, min_token_len: int = 2):
        self.max_token_df = max_token_df
        self.max_addr_df = max_addr_df
        self.min_token_len = min_token_len
        self.total_docs = 0
        
        # Country-partitioned inverted indices: country -> index
        self.country_indices = defaultdict(lambda: {
            'exact_name': defaultdict(list),
            'exact_stripped': defaultdict(list),
            'exact_compact': defaultdict(list),
            'token_name': defaultdict(list),
            'token_addr': defaultdict(list),
            'postal_prefix': defaultdict(list),
            'name_addr_combined': defaultdict(list),
            'token_df': Counter(),
            'token_addr_df': Counter(),
            'token_idf': {},
            'token_addr_idf': {},
            'doc_count': 0
        })
        
        self.token_df = Counter()
        self.token_idf: Dict[str, float] = {}

    def index_target_records(self, target_records: list):
        self.total_docs = len(target_records)
        print(f"Indexing {self.total_docs:,} target records across multi-channel inverted indices...")
        
        # Step 1: Count document frequency
        for idx, rec in enumerate(target_records):
            c_idx = self.country_indices[rec.country]
            c_idx['doc_count'] += 1
            
            seen_name = set(rec.tokens_name)
            for t in seen_name:
                if len(t) >= self.min_token_len:
                    self.token_df[t] += 1
                    c_idx['token_df'][t] += 1
                    
            seen_addr = set(rec.rare_tokens_address)
            for t in seen_addr:
                c_idx['token_addr_df'][t] += 1
                    
        # Compute IDF
        N = float(self.total_docs)
        for t, df in self.token_df.items():
            self.token_idf[t] = math.log((N + 1.0) / (df + 1.0)) + 1.0
            
        for country, c_idx in self.country_indices.items():
            c_N = float(c_idx['doc_count'])
            for t, df in c_idx['token_df'].items():
                c_idx['token_idf'][t] = math.log((c_N + 1.0) / (df + 1.0)) + 1.0
            for t, df in c_idx['token_addr_df'].items():
                c_idx['token_addr_idf'][t] = math.log((c_N + 1.0) / (df + 1.0)) + 1.0
            
        # Step 2: Populate inverted postings
        for idx, rec in enumerate(target_records):
            c_idx = self.country_indices[rec.country]
            
            # 1. Exact Name
            if rec.norm_name and len(rec.norm_name) > 2:
                c_idx['exact_name'][rec.norm_name].append(idx)
                
            # 2. Exact Stripped Name
            if rec.legal_stripped_name and len(rec.legal_stripped_name) > 2:
                c_idx['exact_stripped'][rec.legal_stripped_name].append(idx)
                
            # 3. Compact Name & Compact Stripped Name
            if rec.compact_name and len(rec.compact_name) > 3:
                c_idx['exact_compact'][rec.compact_name].append(idx)
            if rec.compact_stripped_name and len(rec.compact_stripped_name) > 3 and rec.compact_stripped_name != rec.compact_name:
                c_idx['exact_compact'][rec.compact_stripped_name].append(idx)
                
            # 4. Name Tokens
            for t in set(rec.tokens_name):
                df = c_idx['token_df'][t]
                if 1 <= df <= self.max_token_df:
                    c_idx['token_name'][t].append(idx)
                    
            # 5. Address Rare Tokens
            for t in set(rec.rare_tokens_address):
                df = c_idx['token_addr_df'][t]
                if 1 <= df <= self.max_addr_df:
                    c_idx['token_addr'][t].append(idx)
                    
            # 6. Postal Code + Prefix
            if rec.postal_code and rec.norm_name:
                key = (rec.postal_code, rec.norm_name[:3])
                c_idx['postal_prefix'][key].append(idx)
                
            # 7. Name + Address Combined Channel (up to 3 tokens each)
            for nt in rec.tokens_name[:3]:
                for at in rec.rare_tokens_address[:3]:
                    c_idx['name_addr_combined'][(nt, at)].append(idx)
                
        print(f"Indexing complete. Indexed tokens: name={len(self.token_df):,}. Countries indexed: {list(self.country_indices.keys())}")

    def retrieve_candidates(self, query_rec, max_candidates_per_channel: int = 40, top_k_total: int = 75) -> Dict[int, dict]:
        """
        Retrieves candidates for an S1 query record across all channels.
        """
        candidate_info = defaultdict(lambda: {
            'channels': set(),
            'channel_ranks': {},
            'channel_scores': {},
            'token_hits': 0,
            'max_idf': 0.0,
            'sum_idf': 0.0,
        })
        
        target_indices = []
        if query_rec.country and query_rec.country in self.country_indices:
            target_indices.append(self.country_indices[query_rec.country])
        else:
            target_indices = list(self.country_indices.values())
            
        for c_idx in target_indices:
            # 1. Exact Name
            if query_rec.norm_name in c_idx['exact_name']:
                hits = c_idx['exact_name'][query_rec.norm_name][:max_candidates_per_channel]
                for rank, tidx in enumerate(hits, 1):
                    candidate_info[tidx]['channels'].add(RetrievalChannel.EXACT_NAME)
                    candidate_info[tidx]['channel_ranks'][RetrievalChannel.EXACT_NAME] = rank
                    candidate_info[tidx]['channel_scores'][RetrievalChannel.EXACT_NAME] = 1.0

            # 2. Exact Stripped
            if query_rec.legal_stripped_name in c_idx['exact_stripped']:
                hits = c_idx['exact_stripped'][query_rec.legal_stripped_name][:max_candidates_per_channel]
                for rank, tidx in enumerate(hits, 1):
                    candidate_info[tidx]['channels'].add(RetrievalChannel.EXACT_STRIPPED)
                    candidate_info[tidx]['channel_ranks'][RetrievalChannel.EXACT_STRIPPED] = rank
                    candidate_info[tidx]['channel_scores'][RetrievalChannel.EXACT_STRIPPED] = 0.95

            # 3. Compact Name & Compact Stripped
            for c_name in [query_rec.compact_name, query_rec.compact_stripped_name]:
                if c_name and c_name in c_idx['exact_compact']:
                    hits = c_idx['exact_compact'][c_name][:max_candidates_per_channel]
                    for rank, tidx in enumerate(hits, 1):
                        candidate_info[tidx]['channels'].add(RetrievalChannel.EXACT_COMPACT)
                        candidate_info[tidx]['channel_ranks'][RetrievalChannel.EXACT_COMPACT] = rank
                        candidate_info[tidx]['channel_scores'][RetrievalChannel.EXACT_COMPACT] = 0.90

            # 4. Name + Address Combined Channel
            for nt in query_rec.tokens_name[:3]:
                for at in query_rec.rare_tokens_address[:3]:
                    comb_hits = c_idx['name_addr_combined'].get((nt, at), [])
                    for rank, tidx in enumerate(comb_hits[:max_candidates_per_channel], 1):
                        candidate_info[tidx]['channels'].add(RetrievalChannel.NAME_ADDR_COMBINED)
                        if RetrievalChannel.NAME_ADDR_COMBINED not in candidate_info[tidx]['channel_ranks']:
                            candidate_info[tidx]['channel_ranks'][RetrievalChannel.NAME_ADDR_COMBINED] = rank
                            candidate_info[tidx]['channel_scores'][RetrievalChannel.NAME_ADDR_COMBINED] = 0.88

            # 5. Postal Code + Prefix
            if query_rec.postal_code and query_rec.norm_name:
                key = (query_rec.postal_code, query_rec.norm_name[:3])
                if key in c_idx['postal_prefix']:
                    hits = c_idx['postal_prefix'][key][:max_candidates_per_channel]
                    for rank, tidx in enumerate(hits, 1):
                        candidate_info[tidx]['channels'].add(RetrievalChannel.POSTAL_PREFIX)
                        candidate_info[tidx]['channel_ranks'][RetrievalChannel.POSTAL_PREFIX] = rank
                        candidate_info[tidx]['channel_scores'][RetrievalChannel.POSTAL_PREFIX] = 0.85

            # 6. Rare & Informative Name Tokens
            token_idf_dict = c_idx['token_idf']
            q_tokens = [t for t in set(query_rec.tokens_name) if t in token_idf_dict]
            q_tokens.sort(key=lambda t: token_idf_dict[t], reverse=True)
            
            token_candidate_scores = Counter()
            for t in q_tokens:
                idf = token_idf_dict[t]
                if t in c_idx['token_name']:
                    postings = c_idx['token_name'][t]
                    sub_postings = postings[:max_candidates_per_channel * 2]
                    for tidx in sub_postings:
                        token_candidate_scores[tidx] += idf
                        info = candidate_info[tidx]
                        info['token_hits'] += 1
                        info['sum_idf'] += idf
                        if idf > info['max_idf']:
                            info['max_idf'] = idf

            top_token_hits = token_candidate_scores.most_common(max_candidates_per_channel)
            for rank, (tidx, sc) in enumerate(top_token_hits, 1):
                info = candidate_info[tidx]
                info['channels'].add(RetrievalChannel.RARE_NAME_TOKEN)
                info['channel_ranks'][RetrievalChannel.RARE_NAME_TOKEN] = rank
                info['channel_scores'][RetrievalChannel.RARE_NAME_TOKEN] = sc

            # 7. Address Rare Token Channel
            addr_idf_dict = c_idx['token_addr_idf']
            qa_tokens = [t for t in set(query_rec.rare_tokens_address) if t in addr_idf_dict]
            qa_tokens.sort(key=lambda t: addr_idf_dict[t], reverse=True)
            
            addr_candidate_scores = Counter()
            for t in qa_tokens[:5]:
                idf = addr_idf_dict[t]
                if t in c_idx['token_addr']:
                    for tidx in c_idx['token_addr'][t][:max_candidates_per_channel]:
                        addr_candidate_scores[tidx] += idf
                        
            top_addr_hits = addr_candidate_scores.most_common(max_candidates_per_channel)
            for rank, (tidx, sc) in enumerate(top_addr_hits, 1):
                info = candidate_info[tidx]
                info['channels'].add(RetrievalChannel.ADDR_RARE_TOKEN)
                info['channel_ranks'][RetrievalChannel.ADDR_RARE_TOKEN] = rank
                info['channel_scores'][RetrievalChannel.ADDR_RARE_TOKEN] = sc

        # Prune to top_k_total based on composite multi-channel heuristic
        if len(candidate_info) > top_k_total:
            def candidate_priority(item):
                tidx, data = item
                n_ch = len(data['channels'])
                has_exact = 1 if (RetrievalChannel.EXACT_NAME in data['channels'] or 
                                  RetrievalChannel.EXACT_STRIPPED in data['channels'] or
                                  RetrievalChannel.EXACT_COMPACT in data['channels'] or
                                  RetrievalChannel.NAME_ADDR_COMBINED in data['channels']) else 0
                return (has_exact, n_ch, data['sum_idf'], data['max_idf'])
            
            sorted_candidates = sorted(candidate_info.items(), key=candidate_priority, reverse=True)
            candidate_info = dict(sorted_candidates[:top_k_total])

        return candidate_info
