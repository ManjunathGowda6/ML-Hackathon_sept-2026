import os
import sys
sys.path.insert(0, r"D:\Arnav")
import re
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter

from multiview import MultiViewRecord, compute_multiview_pair_features
from retrieval import MultiChannelRetriever
from recall_recovery import RecallRecoveryRetriever, CandidateEvidenceMerger
from features import extract_candidate_pair_vector, FEATURE_COLUMNS
from metrics import compute_entity_macro_f05

TARGETS_FILE = r"D:\Arnav\validation\val_targets_500k.tsv"
SOURCE1_FILE = r"D:\Arnav\validation\val_source1.tsv"
GT_FILE = r"D:\Arnav\validation\val_ground_truth.tsv"
V5_MODEL_PATH = r"D:\Arnav\models\pair_match_model.txt"

def classify_false_negative(r1: MultiViewRecord, r2: MultiViewRecord, ret_info: dict, sc: float) -> str:
    # A. Transliteration / B. Multilingual
    has_non_ascii_1 = any(ord(c) > 127 for c in r1.raw_name)
    has_non_ascii_2 = any(ord(c) > 127 for c in r2.raw_name)
    if has_non_ascii_1 or has_non_ascii_2:
        return "A. transliteration / B. multilingual/script mismatch"
        
    # J. Domain/URL
    if r1.domain_stem or r2.domain_stem:
        if r1.domain_stem == r2.domain_stem or (r1.domain_stem and r1.domain_stem in r2.norm_name) or (r2.domain_stem and r2.domain_stem in r1.norm_name):
            return "J. domain/URL mismatch"
            
    # C. Zero lexical overlap
    toks1 = set(r1.tokens_name)
    toks2 = set(r2.tokens_name)
    if not toks1.intersection(toks2):
        if r1.norm_address == r2.norm_address or (r1.postal_code and r1.postal_code == r2.postal_code):
            return "K. DBA/trade name"
        return "C. zero lexical overlap"
        
    # H. Missing address
    if not r1.norm_address or not r2.norm_address:
        return "H. missing address"
        
    # F. Address abbreviation / G. Address reorder
    if r1.postal_code and r2.postal_code and r1.postal_code == r2.postal_code:
        if r1.house_number and r2.house_number and r1.house_number == r2.house_number:
            return "F. address abbreviation / G. address reorder"
            
    # I. Postal/City/State mismatch
    if r1.postal_code and r2.postal_code and r1.postal_code != r2.postal_code:
        return "I. postal/city/state mismatch"
        
    # M. Word-order variation / L. Legal suffix
    if sorted(toks1) == sorted(toks2):
        return "M. word-order variation"
    if r1.legal_stripped_name == r2.legal_stripped_name:
        return "L. legal suffix variation"
        
    # S. Multi-match ranking error
    if sc >= 0.30:
        return "S. multi-match ranking error"
        
    # D. Rare token failure
    return "D. rare-token failure / E. high-frequency blocking"

def classify_false_positive(r1: MultiViewRecord, r2: MultiViewRecord, sc: float) -> str:
    # Cross-country
    if r1.country != r2.country:
        return "T. cross-country conflict"
        
    # P. Same address different business / Q. Commercial building collision
    addr_match = (r1.norm_address and r1.norm_address == r2.norm_address) or (r1.postal_code and r1.postal_code == r2.postal_code and r1.house_number and r1.house_number == r2.house_number)
    name_toks1 = set(r1.tokens_name)
    name_toks2 = set(r2.tokens_name)
    name_overlap = len(name_toks1.intersection(name_toks2))
    
    if addr_match and name_overlap == 0:
        return "Q. commercial-building collision / P. same-address different-business"
        
    # O. Same name different business / R. Chain/franchise
    if r1.compact_stripped_name == r2.compact_stripped_name or r1.norm_name == r2.norm_name:
        if r1.postal_code and r2.postal_code and r1.postal_code != r2.postal_code:
            return "O. same-name different-business / R. chain/franchise collision"
            
    # Generic high frequency tokens
    generic_terms = {'enterprises', 'trading', 'solutions', 'services', 'company', 'group', 'industries', 'associates'}
    if name_toks1.intersection(generic_terms) and name_overlap <= 1:
        return "E. generic high-frequency token collision"
        
    # S. Multi-match ranking error (loose margin)
    return "S. multi-match ranking error"

def main():
    print("=" * 60)
    print("MANDATORY DEEP FORENSIC ERROR MINING (COMBINED-E SYSTEM)")
    print("=" * 60)
    
    # Load GT
    df_gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)
    df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].fillna('')
    gt_map = {}
    total_true = 0
    for _, row in df_gt.iterrows():
        s1 = row['source1_entity_id'].strip()
        m = {x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()}
        gt_map[s1] = m
        total_true += len(m)
        
    # Load S1
    df_s1 = pd.read_csv(SOURCE1_FILE, sep="\t", dtype=str)
    s1_records = [
        MultiViewRecord(row['entity_id'], row['business_name'], row['business_address'], row['country'])
        for _, row in df_s1.iterrows()
    ]
    s1_map = {r.entity_id: r for r in s1_records}
    
    # Load Targets
    df_tgt = pd.read_csv(TARGETS_FILE, sep="\t", dtype=str)
    tgt_records = [
        MultiViewRecord(row['entity_id'], row['business_name'], row['business_address'], row['country'])
        for _, row in df_tgt.iterrows()
    ]
    tgt_map = {r.entity_id: r for r in tgt_records}
    
    # Build Retrievers
    print("Indexing target catalog...")
    primary = MultiChannelRetriever(max_token_df=12000)
    primary.index_target_records(tgt_records)
    recovery = RecallRecoveryRetriever(max_token_df=12000)
    recovery.index_target_records(tgt_records)
    
    name_freq_dict = Counter(r.norm_name for r in tgt_records if r.norm_name)
    addr_freq_dict = Counter(r.norm_address for r in tgt_records if r.norm_address)
    
    # Load Model
    model = lgb.Booster(model_file=V5_MODEL_PATH)
    
    # Run Combined E
    print("Running Combined-E candidate retrieval & scoring...")
    feature_rows = []
    pair_meta = []
    cand_by_s1 = defaultdict(list)
    
    for r1 in s1_records:
        p_cands = primary.retrieve_candidates(r1, max_candidates_per_channel=40, top_k_total=75)
        r_cands = recovery.retrieve_recovery_candidates(r1, max_per_channel=15, recovery_budget=15)
        merged = CandidateEvidenceMerger.merge(p_cands, r_cands, max_total_budget=80)
        
        for tidx, ret_info in merged.items():
            r2 = tgt_records[tidx]
            f_vec = extract_candidate_pair_vector(r1, r2, ret_info, name_freq_dict, addr_freq_dict, primary.token_idf)
            feature_rows.append(f_vec)
            pair_meta.append((r1.entity_id, r2.entity_id, r2, ret_info, f_vec))
            
    X = np.array(feature_rows, dtype=np.float32)
    raw_scores = model.predict(X, num_threads=10)
    
    for i, sc in enumerate(raw_scores):
        s1_id, t_id, r2, ret_info, f_vec = pair_meta[i]
        cand_by_s1[s1_id].append({
            'target_id': t_id,
            'raw_score': float(sc),
            'ret_info': ret_info,
            'is_source_2': 1.0 if t_id.startswith('S2-') else 0.0,
            'is_source_3': 1.0 if t_id.startswith('S3-') else 0.0,
            'postal_conflict': f_vec[FEATURE_COLUMNS.index('postal_conflict')],
            'country_conflict': f_vec[FEATURE_COLUMNS.index('country_conflict')],
            'house_num_conflict': f_vec[FEATURE_COLUMNS.index('house_num_conflict')],
            'name_exact_match': f_vec[FEATURE_COLUMNS.index('name_exact_match')],
            'name_ratio': f_vec[FEATURE_COLUMNS.index('name_ratio')],
            'addr_token_set_ratio': f_vec[FEATURE_COLUMNS.index('addr_token_set_ratio')],
        })
        
    # Decision Layer
    min_top = 0.36
    multi_th = 0.34
    max_gap = 0.12
    s2_pen = 0.04
    max_k_cap = 8
    
    predictions_map = {}
    pair_scores = {}
    
    for r1 in s1_records:
        cands = cand_by_s1.get(r1.entity_id, [])
        if not cands:
            predictions_map[r1.entity_id] = set()
            continue
            
        scored = []
        for c in cands:
            sc = c['raw_score']
            if c['is_source_2'] > 0.5: sc -= s2_pen
            if c['country_conflict'] > 0.5: sc -= 0.50
            if c['postal_conflict'] > 0.5 and c['name_exact_match'] < 0.5: sc -= 0.25
            if c['house_num_conflict'] > 0.5 and c['name_exact_match'] < 0.5: sc -= 0.15
            if c.get('addr_token_set_ratio', 0) > 85 and c.get('name_ratio', 0) < 40: sc -= 0.20
            scored.append({'id': c['target_id'], 'sc': sc, 'raw': c['raw_score'], 'ret_info': c['ret_info'], 'meta': c})
            pair_scores[(r1.entity_id, c['target_id'])] = sc
            
        scored.sort(key=lambda x: x['sc'], reverse=True)
        top = scored[0]
        if top['sc'] < min_top:
            predictions_map[r1.entity_id] = set()
            continue
            
        accepted = [top['id']]
        top_sc = top['sc']
        for cand in scored[1:]:
            c_sc = cand['sc']
            gap = top_sc - c_sc
            meta = cand['meta']
            if (c_sc >= multi_th and gap <= max_gap and meta['country_conflict'] < 0.5 and meta['postal_conflict'] < 0.5):
                accepted.append(cand['id'])
                if len(accepted) >= max_k_cap: break
            elif gap > max_gap + 0.08: break
            
        predictions_map[r1.entity_id] = set(accepted)
        
    # Extract FN and FP
    fn_pairs = []
    fp_pairs = []
    
    for s1_id, true_tgts in gt_map.items():
        preds = predictions_map.get(s1_id, set())
        for t in true_tgts:
            if t not in preds:
                fn_pairs.append((s1_id, t))
        for t in preds:
            if t not in true_tgts:
                fp_pairs.append((s1_id, t))
                
    print(f"\nTotal Ground Truth Links: {total_true:,}")
    print(f"Total True Positives (TP): {total_true - len(fn_pairs):,}")
    print(f"Total False Negatives (FN): {len(fn_pairs):,}")
    print(f"Total False Positives (FP): {len(fp_pairs):,}")
    
    # Classify FN
    fn_categories = Counter()
    for s1_id, t_id in fn_pairs:
        r1 = s1_map[s1_id]
        r2 = tgt_map.get(t_id)
        if not r2:
            fn_categories["T. other / unindexed target"] += 1
            continue
        sc = pair_scores.get((s1_id, t_id), 0.0)
        ret_info = {}
        for c in cand_by_s1.get(s1_id, []):
            if c['target_id'] == t_id:
                ret_info = c['ret_info']
                break
        cat = classify_false_negative(r1, r2, ret_info, sc)
        fn_categories[cat] += 1
        
    # Classify FP
    fp_categories = Counter()
    for s1_id, t_id in fp_pairs:
        r1 = s1_map[s1_id]
        r2 = tgt_map.get(t_id)
        if not r2:
            fp_categories["T. other"] += 1
            continue
        sc = pair_scores.get((s1_id, t_id), 0.0)
        cat = classify_false_positive(r1, r2, sc)
        fp_categories[cat] += 1
        
    print("\n" + "=" * 70)
    print("FALSE NEGATIVE BREAKDOWN (Total: %d):" % len(fn_pairs))
    print("=" * 70)
    for cat, cnt in fn_categories.most_common():
        pct = (cnt / len(fn_pairs)) * 100
        print(f"  {cat:<52s}: {cnt:4d} ({pct:5.1f}%)")
        
    print("\n" + "=" * 70)
    print("FALSE POSITIVE BREAKDOWN (Total: %d):" % len(fp_pairs))
    print("=" * 70)
    for cat, cnt in fp_categories.most_common():
        pct = (cnt / len(fp_pairs)) * 100
        print(f"  {cat:<52s}: {cnt:4d} ({pct:5.1f}%)")
        
    # Save Report
    rep_path = r"D:\Arnav\reports\DEEP_FORENSIC_ERROR_MINING.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Deep Forensic Error Mining Report (Combined-E System)\n\n")
        f.write(f"- **Total True Links**: {total_true:,}\n")
        f.write(f"- **False Negatives (FN)**: {len(fn_pairs):,} ({len(fn_pairs)/total_true*100:.1f}%)\n")
        f.write(f"- **False Positives (FP)**: {len(fp_pairs):,}\n\n")
        
        f.write("## 1. False Negative Categories\n\n")
        f.write("| Error Category | Count | Percentage |\n|---|---|---|\n")
        for cat, cnt in fn_categories.most_common():
            f.write(f"| {cat} | {cnt:,} | {cnt/len(fn_pairs)*100:.1f}% |\n")
            
        f.write("\n## 2. False Positive Categories\n\n")
        f.write("| Error Category | Count | Percentage |\n|---|---|---|\n")
        for cat, cnt in fp_categories.most_common():
            f.write(f"| {cat} | {cnt:,} | {cnt/len(fp_pairs)*100:.1f}% |\n")
            
    print(f"\nForensic report written to {rep_path}")

if __name__ == "__main__":
    main()
