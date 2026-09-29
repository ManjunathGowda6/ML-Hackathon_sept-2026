import os
import sys
sys.path.insert(0, r"D:\Arnav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict, Counter

from multiview import MultiViewRecord
from retrieval import MultiChannelRetriever, RetrievalChannel
from recall_recovery import RecallRecoveryRetriever, CandidateEvidenceMerger
from features import extract_candidate_pair_vector, FEATURE_COLUMNS
from graph_ranker import EntityMatchDecider
from metrics import compute_entity_macro_f05

TARGETS_FILE = r"D:\Arnav\validation\val_targets_500k.tsv"
SOURCE1_FILE = r"D:\Arnav\validation\val_source1.tsv"
GT_FILE = r"D:\Arnav\validation\val_ground_truth.tsv"
V5_MODEL_PATH = r"D:\Arnav\models\pair_match_model.txt"

def load_data():
    print("Loading Ground Truth...")
    df_gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)
    df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].fillna('')
    gt_map = {}
    total_true = 0
    for _, row in df_gt.iterrows():
        s1 = row['source1_entity_id'].strip()
        m = {x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()}
        gt_map[s1] = m
        total_true += len(m)
        
    print(f"Loading S1 queries ({SOURCE1_FILE})...")
    df_s1 = pd.read_csv(SOURCE1_FILE, sep="\t", dtype=str)
    s1_records = [
        MultiViewRecord(row['entity_id'], row['business_name'], row['business_address'], row['country'])
        for _, row in df_s1.iterrows()
    ]
    
    print(f"Loading Target Catalog ({TARGETS_FILE})...")
    df_tgt = pd.read_csv(TARGETS_FILE, sep="\t", dtype=str)
    tgt_records = [
        MultiViewRecord(row['entity_id'], row['business_name'], row['business_address'], row['country'])
        for _, row in df_tgt.iterrows()
    ]
    
    return gt_map, total_true, s1_records, tgt_records

def run_experiment(exp_name: str,
                   recovery_mode: str, # 'none', 'A_high_freq', 'B_zero_token', 'C_ranking', 'E_combined'
                   s1_records,
                   tgt_records,
                   gt_map,
                   total_true,
                   primary_retriever,
                   recovery_retriever,
                   name_freq_dict,
                   addr_freq_dict,
                   model):
    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT: {exp_name} (Recovery Mode: {recovery_mode})")
    print("=" * 60)
    t0 = time.time()
    
    retrieved_hits = 0
    feature_rows = []
    pair_meta = []
    
    t_ret_start = time.time()
    for r1 in s1_records:
        true_tids = gt_map.get(r1.entity_id, set())
        
        # 1. Primary V5 retrieval
        p_cands = primary_retriever.retrieve_candidates(r1, max_candidates_per_channel=40, top_k_total=75)
        
        # 2. Targeted Recovery based on mode
        merged_cands = dict(p_cands)
        
        if recovery_mode == 'A_high_freq':
            # Recovery A: Only if primary set is small (< 25)
            if len(p_cands) < 25:
                r_cands = recovery_retriever.retrieve_recovery_candidates(
                    r1, max_per_channel=15, recovery_budget=15
                )
                # Keep only rare token pair / name-addr cross
                filtered_r = {k: v for k, v in r_cands.items() if v.get('candidate_category') in ('B', 'C')}
                merged_cands = CandidateEvidenceMerger.merge(p_cands, filtered_r, max_total_budget=80)
                
        elif recovery_mode == 'B_zero_token':
            # Recovery B: Transliteration + Domain stem + Sorted stripped name
            r_cands = recovery_retriever.retrieve_recovery_candidates(
                r1, max_per_channel=15, recovery_budget=15
            )
            filtered_r = {k: v for k, v in r_cands.items() if v.get('candidate_category') in ('D', 'A')}
            merged_cands = CandidateEvidenceMerger.merge(p_cands, filtered_r, max_total_budget=80)
            
        elif recovery_mode == 'E_combined':
            # Combined targeted recovery: tightly budgeted (max 15 recovery cands)
            r_cands = recovery_retriever.retrieve_recovery_candidates(
                r1, max_per_channel=15, recovery_budget=15
            )
            merged_cands = CandidateEvidenceMerger.merge(p_cands, r_cands, max_total_budget=80)
            
        # Feature extraction using exact V5 47-feature extractor
        for tidx, ret_info in merged_cands.items():
            r2 = tgt_records[tidx]
            if r2.entity_id in true_tids:
                retrieved_hits += 1
            f_vec = extract_candidate_pair_vector(r1, r2, ret_info, name_freq_dict, addr_freq_dict, primary_retriever.token_idf)
            feature_rows.append(f_vec)
            pair_meta.append((r1.entity_id, r2.entity_id, r2, ret_info, f_vec))
            
    t_ret = time.time() - t_ret_start
    cand_recall = (retrieved_hits / total_true) * 100 if total_true > 0 else 0.0
    print(f"Candidate Generation: {len(feature_rows):,} pairs in {t_ret:.2f}s | Candidate Recall: {cand_recall:.2f}%")
    
    # 3. V5 Scorer Inference
    t_score_start = time.time()
    X = np.array(feature_rows, dtype=np.float32)
    raw_scores = model.predict(X, num_threads=10)
    t_score = time.time() - t_score_start
    print(f"Scoring: {len(raw_scores):,} pairs scored in {t_score:.2f}s ({len(raw_scores)/t_score:,.0f} pairs/sec)")
    
    # Group scored candidates by S1 entity
    cand_by_s1 = defaultdict(list)
    for i, sc in enumerate(raw_scores):
        s1_id, t_id, r2, ret_info, f_vec = pair_meta[i]
        cand_by_s1[s1_id].append({
            'target_id': t_id,
            'raw_score': float(sc),
            'is_source_2': 1.0 if t_id.startswith('S2-') else 0.0,
            'is_source_3': 1.0 if t_id.startswith('S3-') else 0.0,
            'postal_conflict': f_vec[FEATURE_COLUMNS.index('postal_conflict')],
            'country_conflict': f_vec[FEATURE_COLUMNS.index('country_conflict')],
            'house_num_conflict': f_vec[FEATURE_COLUMNS.index('house_num_conflict')],
            'name_exact_match': f_vec[FEATURE_COLUMNS.index('name_exact_match')],
            'channel_hit_count': f_vec[FEATURE_COLUMNS.index('channel_hit_count')],
            'name_ratio': f_vec[FEATURE_COLUMNS.index('name_ratio')],
            'addr_token_set_ratio': f_vec[FEATURE_COLUMNS.index('addr_token_set_ratio')],
        })
        
    # 4. Ranking & Decision Policy
    # In V5 baseline: min_top=0.34, multi=0.32, gap=0.15, s2_pen=0.03
    # In Improvement C / E: precision-protective multi-match ranking + false-positive protection
    if recovery_mode in ('C_ranking', 'E_combined'):
        min_top = 0.36
        multi_th = 0.34
        max_gap = 0.12
        s2_pen = 0.04
        max_k_cap = 8
    else:
        min_top = 0.34
        multi_th = 0.32
        max_gap = 0.15
        s2_pen = 0.03
        max_k_cap = 99
        
    predictions_map = {}
    for r1 in s1_records:
        cands = cand_by_s1.get(r1.entity_id, [])
        if not cands:
            predictions_map[r1.entity_id] = set()
            continue
            
        scored = []
        for c in cands:
            sc = c['raw_score']
            if c['is_source_2'] > 0.5:
                sc -= s2_pen
            if c['country_conflict'] > 0.5:
                sc -= 0.50
            if c['postal_conflict'] > 0.5 and c['name_exact_match'] < 0.5:
                sc -= 0.25
            if c['house_num_conflict'] > 0.5 and c['name_exact_match'] < 0.5:
                sc -= 0.15
                
            # False-positive protection: Same address + weak name (< 0.45 name ratio)
            if recovery_mode in ('C_ranking', 'E_combined'):
                if c.get('addr_token_set_ratio', 0) > 85 and c.get('name_ratio', 0) < 40:
                    sc -= 0.20 # Commercial building collision protection
                    
            scored.append({'id': c['target_id'], 'sc': sc, 'meta': c})
            
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
            if (c_sc >= multi_th and gap <= max_gap and
                meta['country_conflict'] < 0.5 and
                meta['postal_conflict'] < 0.5):
                accepted.append(cand['id'])
                if len(accepted) >= max_k_cap:
                    break
            elif gap > max_gap + 0.08:
                break
                
        predictions_map[r1.entity_id] = set(accepted)
        
    metrics = compute_entity_macro_f05(gt_map, predictions_map)
    metrics['candidate_recall'] = cand_recall
    metrics['t_retrieval'] = t_ret
    metrics['t_scoring'] = t_score
    metrics['t_total'] = time.time() - t0
    
    print(f"Results for {exp_name}:")
    print(f"  Entity Macro F0.5 : {metrics['entity_macro_f05']:.4f}")
    print(f"  Entity Precision  : {metrics['entity_macro_precision']:.4f}")
    print(f"  Entity Recall     : {metrics['entity_macro_recall']:.4f}")
    print(f"  Pair Precision    : {metrics['pair_precision']:.4f}")
    print(f"  Pair Recall       : {metrics['pair_recall']:.4f}")
    print(f"  Candidate Recall  : {metrics['candidate_recall']:.2f}%")
    print(f"  Singleton Acc     : {metrics['singleton_accuracy']*100:.2f}%")
    print(f"  TP / FP / FN      : {metrics['total_tp']} / {metrics['total_fp']} / {metrics['total_fn']}")
    print(f"  Mean Matches/S1   : {metrics['pred_matches_mean']:.2f}")
    print(f"  Runtime (Total)   : {metrics['t_total']:.2f}s (Ret: {t_ret:.2f}s, Score: {t_score:.2f}s)")
    
    return metrics

def main():
    gt_map, total_true, s1_records, tgt_records = load_data()
    
    print("\nBuilding Primary & Secondary Retrievers...")
    t_idx = time.time()
    primary = MultiChannelRetriever(max_token_df=12000)
    primary.index_target_records(tgt_records)
    
    recovery = RecallRecoveryRetriever(max_token_df=12000)
    recovery.index_target_records(tgt_records)
    print(f"Indexing completed in {time.time() - t_idx:.2f}s.")
    
    name_freq_dict = Counter(r.norm_name for r in tgt_records if r.norm_name)
    addr_freq_dict = Counter(r.norm_address for r in tgt_records if r.norm_address)
    
    print(f"Loading V5 Model from {V5_MODEL_PATH}...")
    model = lgb.Booster(model_file=V5_MODEL_PATH)
    
    experiments = [
        ("V5_BASELINE", "none"),
        ("V5_PLUS_REC_A_HIGH_FREQ", "A_high_freq"),
        ("V5_PLUS_REC_B_ZERO_TOKEN", "B_zero_token"),
        ("V5_PLUS_REC_C_RANKING", "C_ranking"),
        ("V5_PLUS_COMBINED_E", "E_combined")
    ]
    
    results = {}
    for name, mode in experiments:
        res = run_experiment(name, mode, s1_records, tgt_records, gt_map, total_true,
                             primary, recovery, name_freq_dict, addr_freq_dict, model)
        results[name] = res
        
    print("\n" + "=" * 80)
    print("V5-ANCHORED ABLATION MATRIX SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<26} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Pair Prec':<9} | {'Cand Rec':<8} | {'FP':<5} | {'FN':<5}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<26} | {r['entity_macro_f05']:<10.4f} | {r['entity_macro_precision']:<9.4f} | {r['entity_macro_recall']:<8.4f} | {r['pair_precision']:<9.4f} | {r['candidate_recall']:<7.2f}% | {r['total_fp']:<5d} | {r['total_fn']:<5d}")
    print("=" * 80)

if __name__ == "__main__":
    main()
