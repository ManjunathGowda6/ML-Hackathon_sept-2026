import numpy as np
from multiview import MultiViewRecord, compute_multiview_pair_features
from retrieval import RetrievalChannel
from recall_recovery import RecoveryChannel

FEATURE_COLUMNS = [
    # Name features
    'name_exact_match',
    'name_compact_exact',
    'name_stripped_exact',
    'name_sorted_exact',
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_stripped_ratio',
    'name_token_jaccard',
    'name_token_containment',
    'name_weighted_overlap',
    'name_max_overlap_idf',
    'name_len_diff',
    'name_len_ratio',
    'legal_suffix_compatible',
    'legal_suffix_conflict',
    
    # Address features
    'addr_exact_match',
    'addr_compact_exact',
    'addr_sorted_exact',
    'addr_ratio',
    'addr_token_set_ratio',
    'addr_partial_ratio',
    'addr_token_jaccard',
    'addr_token_containment',
    
    # Address component & conflict features
    'postal_exact_match',
    'postal_conflict',
    'house_num_match',
    'house_num_conflict',
    'country_match',
    'country_conflict',
    
    # Composite features
    'name_x_addr_ratio',
    'name_x_addr_jaccard',
    
    # Source indicator
    'is_source_2',
    'is_source_3',
    
    # Retrieval channel features
    'channel_hit_count',
    'hit_exact_name',
    'hit_exact_stripped',
    'hit_exact_compact',
    'hit_rare_token',
    'hit_postal_prefix',
    'retrieval_best_rank',
    'retrieval_max_idf',
    'retrieval_sum_idf',
    'retrieval_token_hits',
    
    # Frequency / Rarity features
    'target_name_frequency',
    'target_addr_frequency',
]

UPGRADED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    # Transliteration evidence
    'translit_name_ratio',
    'translit_token_sort_ratio',
    'translit_name_jaccard',
    
    # Domain / Handle evidence
    'domain_stem_match',
    
    # Character n-gram evidence
    'char_3gram_jaccard',
    
    # Missing vs Conflict address flags
    'missing_address_flag',
    'missing_postal_flag',
    'missing_house_flag',
    
    # Address anchor & Cross-field support
    'addr_anchor_match',
    'cross_field_support',
    
    # Candidate source & Category evidence
    'is_recovery_candidate',
    'is_primary_candidate',
    'cat_A',
    'cat_B',
    'cat_C',
    'cat_D',
    'cat_E',
    
    # Specific Recovery Channel Hit Indicators
    'hit_domain_handle',
    'hit_transliteration',
    'hit_addr_anchor',
    'hit_rare_token_pair',
    'hit_char_ngram',
    'hit_token_order_invariant',
    'hit_compact_signature',
]

def extract_candidate_pair_vector(r1: MultiViewRecord, r2: MultiViewRecord, ret_info: dict,
                                  name_freq_dict=None, addr_freq_dict=None, idf_dict=None) -> list:
    """
    Extracts baseline 47-feature vector for a candidate pair (r1, r2).
    """
    mv_feats = compute_multiview_pair_features(r1, r2, idf_dict=idf_dict, cand_evidence=ret_info)
    
    channels = ret_info.get('channels', set())
    ranks = ret_info.get('channel_ranks', {})
    mv_feats['channel_hit_count'] = float(len(channels))
    mv_feats['hit_exact_name'] = 1.0 if RetrievalChannel.EXACT_NAME in channels else 0.0
    mv_feats['hit_exact_stripped'] = 1.0 if RetrievalChannel.EXACT_STRIPPED in channels else 0.0
    mv_feats['hit_exact_compact'] = 1.0 if RetrievalChannel.EXACT_COMPACT in channels else 0.0
    mv_feats['hit_rare_token'] = 1.0 if RetrievalChannel.RARE_NAME_TOKEN in channels else 0.0
    mv_feats['hit_postal_prefix'] = 1.0 if RetrievalChannel.POSTAL_PREFIX in channels else 0.0
    
    min_rank = min(ranks.values()) if ranks else 999.0
    mv_feats['retrieval_best_rank'] = float(min_rank)
    mv_feats['retrieval_max_idf'] = float(ret_info.get('max_idf', 0.0))
    mv_feats['retrieval_sum_idf'] = float(ret_info.get('sum_idf', 0.0))
    mv_feats['retrieval_token_hits'] = float(ret_info.get('token_hits', 0.0))
    
    if name_freq_dict and r2.norm_name in name_freq_dict:
        mv_feats['target_name_frequency'] = float(name_freq_dict[r2.norm_name])
    else:
        mv_feats['target_name_frequency'] = 1.0
        
    if addr_freq_dict and r2.norm_address in addr_freq_dict:
        mv_feats['target_addr_frequency'] = float(addr_freq_dict[r2.norm_address])
    else:
        mv_feats['target_addr_frequency'] = 1.0
        
    return [mv_feats[col] for col in FEATURE_COLUMNS]


def extract_upgraded_pair_vector(r1: MultiViewRecord, r2: MultiViewRecord, ret_info: dict,
                                 name_freq_dict=None, addr_freq_dict=None, idf_dict=None) -> list:
    """
    Extracts upgraded feature vector for a candidate pair (r1, r2) including all recovery features.
    """
    mv_feats = compute_multiview_pair_features(r1, r2, idf_dict=idf_dict, cand_evidence=ret_info)
    
    channels = ret_info.get('channels', set())
    ranks = ret_info.get('channel_ranks', {})
    mv_feats['channel_hit_count'] = float(len(channels))
    mv_feats['hit_exact_name'] = 1.0 if RetrievalChannel.EXACT_NAME in channels else 0.0
    mv_feats['hit_exact_stripped'] = 1.0 if RetrievalChannel.EXACT_STRIPPED in channels else 0.0
    mv_feats['hit_exact_compact'] = 1.0 if RetrievalChannel.EXACT_COMPACT in channels else 0.0
    mv_feats['hit_rare_token'] = 1.0 if RetrievalChannel.RARE_NAME_TOKEN in channels else 0.0
    mv_feats['hit_postal_prefix'] = 1.0 if RetrievalChannel.POSTAL_PREFIX in channels else 0.0
    
    min_rank = min(ranks.values()) if ranks else 999.0
    mv_feats['retrieval_best_rank'] = float(min_rank)
    mv_feats['retrieval_max_idf'] = float(ret_info.get('max_idf', 0.0))
    mv_feats['retrieval_sum_idf'] = float(ret_info.get('sum_idf', 0.0))
    mv_feats['retrieval_token_hits'] = float(ret_info.get('token_hits', 0.0))
    
    if name_freq_dict and r2.norm_name in name_freq_dict:
        mv_feats['target_name_frequency'] = float(name_freq_dict[r2.norm_name])
    else:
        mv_feats['target_name_frequency'] = 1.0
        
    if addr_freq_dict and r2.norm_address in addr_freq_dict:
        mv_feats['target_addr_frequency'] = float(addr_freq_dict[r2.norm_address])
    else:
        mv_feats['target_addr_frequency'] = 1.0

    # Specific recovery channel hit indicators
    mv_feats['hit_domain_handle'] = 1.0 if RecoveryChannel.DOMAIN_HANDLE in channels else 0.0
    mv_feats['hit_transliteration'] = 1.0 if RecoveryChannel.TRANSLITERATION in channels else 0.0
    mv_feats['hit_addr_anchor'] = 1.0 if RecoveryChannel.ADDRESS_ANCHOR in channels else 0.0
    mv_feats['hit_rare_token_pair'] = 1.0 if RecoveryChannel.RARE_TOKEN_PAIR in channels else 0.0
    mv_feats['hit_char_ngram'] = 1.0 if RecoveryChannel.CHAR_NGRAM in channels else 0.0
    mv_feats['hit_token_order_invariant'] = 1.0 if RecoveryChannel.TOKEN_ORDER_INVARIANT in channels else 0.0
    mv_feats['hit_compact_signature'] = 1.0 if RecoveryChannel.COMPACT_SIGNATURE in channels else 0.0

    return [mv_feats[col] for col in UPGRADED_FEATURE_COLUMNS]
