import os
import sys
sys.path.insert(0, r"D:\Arnav")
sys.stdout.reconfigure(line_buffering=True)
import csv
import time
import gc
import psutil
import sqlite3
import hashlib
import torch
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter

from scripts.multiview import MultiViewRecord
from scripts.features import extract_upgraded_pair_vector, UPGRADED_FEATURE_COLUMNS

DATA_DIR = r"D:\dataset\student_resource\dataset\test"
OUTPUT_DIR = r"D:\Arnav\output_v2_fresh"
CACHE_DIR = r"D:\Arnav\cache"
MODELS_DIR = r"D:\Arnav\models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DB_SOURCE = os.path.join(CACHE_DIR, "full_test_pipeline.db")
DB_FRESH = os.path.join(CACHE_DIR, "fresh_v2_pipeline.db")
MODEL_PATH = os.path.join(MODELS_DIR, "pair_match_model_v2.txt")
TEST_S1_FILE = os.path.join(DATA_DIR, "test_source1.tsv")

OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

TOTAL_REQUIRED_S1 = 1732544
OLD_MATCHING_SHA = "908821E3393B7493992E8A662F186BC0F63FD3FE86A5379422440A9F113BDAC2"

def verify_gpu_hardware():
    print("=" * 60, flush=True)
    print("GPU HARDWARE & CUDA VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    assert torch.cuda.is_available(), "CRITICAL: CUDA is not available on this machine!"
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"CUDA Available: True", flush=True)
    print(f"Device Name: {props.name}", flush=True)
    print(f"CUDA Version (torch): {torch.version.cuda}", flush=True)
    print(f"Compute Capability: {props.major}.{props.minor}", flush=True)
    print(f"Total VRAM: {props.total_memory / (1024**2):.1f} MB", flush=True)
    
    # Real CUDA tensor execution benchmark
    t0 = time.perf_counter()
    x = torch.randn(4096, 71, device=device)
    w = torch.randn(71, 64, device=device)
    b = torch.randn(64, device=device)
    y = torch.matmul(x, w) + b
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"Actual CUDA Tensor Benchmark (batch 4096): {(t1-t0)*1000:.3f}ms | VRAM Allocated: {alloc_mb:.2f} MB", flush=True)
    print("GPU Hardware verification PASSED!\n", flush=True)

def init_fresh_db():
    conn = sqlite3.connect(DB_FRESH, timeout=120.0)
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA cache_size = 100000;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preds (
            s1_id TEXT PRIMARY KEY,
            matched TEXT,
            candidates TEXT
        );
    """)
    conn.commit()
    return conn

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

def run_country_inference(country: str, model: lgb.Booster):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE: FRESH V2 FULL-TEST INFERENCE FOR {country}", flush=True)
    print("=" * 60, flush=True)
    t_stage_start = time.time()
    
    fresh_conn = init_fresh_db()
    fresh_cur = fresh_conn.cursor()
    fresh_cur.execute("SELECT s1_id FROM preds;")
    done_ids = {r[0] for r in fresh_cur.fetchall()}
    print(f"Already completed in fresh DB: {len(done_ids):,} entities.", flush=True)
    
    # 1. Load S1 queries for this country from test_source1.tsv
    print(f"Scanning test_source1.tsv for {country} entities...", flush=True)
    s1_dict = {}
    with open(TEST_S1_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r and len(r) >= 4 and r[3].strip().upper() == country:
                eid = r[0].strip()
                s1_dict[eid] = MultiViewRecord(eid, r[1], r[2], r[3])
                
    total_country_s1 = len(s1_dict)
    print(f"Total S1 entities for {country}: {total_country_s1:,}", flush=True)
    
    remaining_eids = [eid for eid in s1_dict if eid not in done_ids]
    print(f"Remaining S1 entities to process for {country}: {len(remaining_eids):,}", flush=True)
    
    if not remaining_eids:
        print(f"Country {country} already 100% processed in fresh DB! Skipping to next.", flush=True)
        fresh_conn.close()
        return
        
    # 2. Fetch candidate lists for remaining entities from full_test_pipeline.db
    print(f"Fetching candidate sets for {country} from {DB_SOURCE}...", flush=True)
    t_cand_fetch = time.time()
    src_conn = sqlite3.connect(DB_SOURCE, timeout=120.0)
    src_cur = src_conn.cursor()
    
    s1_cands_map = {}
    all_needed_target_ids = set()
    
    SUB = 25000
    for s_idx in range(0, len(remaining_eids), SUB):
        chunk = remaining_eids[s_idx : s_idx + SUB]
        placeholders = ",".join("?" for _ in chunk)
        src_cur.execute(f"SELECT s1_id, candidates FROM preds WHERE s1_id IN ({placeholders});", chunk)
        for row in src_cur.fetchall():
            s1_id = row[0]
            cand_list = [x.strip() for x in row[1].split(",") if x.strip()]
            s1_cands_map[s1_id] = cand_list
            all_needed_target_ids.update(cand_list)
            
    src_conn.close()
    print(f"Fetched candidates for {len(s1_cands_map):,} queries in {time.time() - t_cand_fetch:.1f}s. Unique targets needed: {len(all_needed_target_ids):,}", flush=True)
    
    # 3. Stream only needed target records from test_source2.tsv and test_source3.tsv into memory
    print(f"Streaming target catalog for {country} into memory...", flush=True)
    t_tgt_start = time.time()
    target_dict = {}
    
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join(DATA_DIR, fname)
        c_found = 0
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                if r and len(r) >= 4 and r[0].strip() in all_needed_target_ids:
                    tid = r[0].strip()
                    target_dict[tid] = MultiViewRecord(tid, r[1], r[2], r[3])
                    c_found += 1
        print(f"  Found {c_found:,} matching targets in {fname} in {time.time() - t_tgt_start:.1f}s.", flush=True)
        
    print(f"Total targets loaded in memory: {len(target_dict):,} (Memory: {psutil.Process().memory_info().rss / (1024**2):.1f} MB)", flush=True)
    
    # Pre-calculate frequency dictionaries and token IDF for this country
    print("Computing target frequency and IDF dictionaries...", flush=True)
    name_freq_dict = Counter(r.norm_name for r in target_dict.values() if r.norm_name)
    addr_freq_dict = Counter(r.norm_address for r in target_dict.values() if r.norm_address)
    token_df = Counter()
    for r in target_dict.values():
        for t in r.tokens_name:
            token_df[t] += 1
    N = float(len(target_dict))
    token_idf = defaultdict(lambda: 2.0)
    for t, df in token_df.items():
        token_idf[t] = np.log((N + 1.0) / (df + 1.0)) + 1.0
        
    # Feature column indices for fast lookups
    idx_s2 = UPGRADED_FEATURE_COLUMNS.index('is_source_2')
    idx_country_conf = UPGRADED_FEATURE_COLUMNS.index('country_conflict')
    idx_postal_conf = UPGRADED_FEATURE_COLUMNS.index('postal_conflict')
    idx_house_conf = UPGRADED_FEATURE_COLUMNS.index('house_num_conflict')
    idx_exact_name = UPGRADED_FEATURE_COLUMNS.index('name_exact_match')
    idx_cat_E = UPGRADED_FEATURE_COLUMNS.index('cat_E')
    idx_name_ratio = UPGRADED_FEATURE_COLUMNS.index('name_ratio')
    idx_addr_ratio = UPGRADED_FEATURE_COLUMNS.index('addr_token_set_ratio')
    
    # 4. Process queries in batches
    BATCH_SIZE = 3000
    n_processed = 0
    total_to_process = len(remaining_eids)
    t_query_start = time.time()
    last_report_time = time.time()
    
    print(f"\nScoring {total_to_process:,} {country} entities with validated V2 model and calibrated decider...", flush=True)
    
    for b_start in range(0, total_to_process, BATCH_SIZE):
        batch_eids = remaining_eids[b_start : b_start + BATCH_SIZE]
        feature_rows = []
        pair_meta = []
        
        # Step A: Extract upgraded feature vectors
        for eid in batch_eids:
            r1 = s1_dict[eid]
            cands = s1_cands_map.get(eid, [])
            for rank, tid in enumerate(cands, 1):
                if tid in target_dict:
                    r2 = target_dict[tid]
                    ret_info = {
                        'channels': {'exact_name'} if r1.norm_name == r2.norm_name else {'token_hit'},
                        'channel_ranks': {'exact_name': rank},
                        'channel_scores': {'exact_name': 1.0 / rank},
                        'channel_count': 2 if r1.compact_name == r2.compact_name else 1,
                        'best_retrieval_rank': rank,
                        'best_retrieval_score': 1.0 / rank,
                        'is_primary': 1,
                        'is_recovery': 1 if rank > 20 else 0,
                        'candidate_category': 'D' if r1.compact_name == r2.compact_name else ('B' if rank <= 10 else 'A')
                    }
                    fv = extract_upgraded_pair_vector(r1, r2, ret_info, name_freq_dict, addr_freq_dict, token_idf)
                    feature_rows.append(fv)
                    pair_meta.append((eid, tid, fv))
                    
        # Step B: Score features using trained V2 model
        if feature_rows:
            X = np.array(feature_rows, dtype=np.float32)
            raw_scores = model.predict(X, num_threads=10)
        else:
            raw_scores = []
            
        by_s1 = defaultdict(list)
        for i, sc in enumerate(raw_scores):
            eid, tid, fv = pair_meta[i]
            by_s1[eid].append({'id': tid, 'sc': float(sc), 'f': fv})
            
        # Step C: Exact Validated V2 Decision Policy & Calibrated Gate (K <= 8)
        batch_db_rows = []
        for eid in batch_eids:
            cands = by_s1.get(eid, [])
            all_cands_list = list(s1_cands_map.get(eid, []))
            
            if not cands:
                batch_db_rows.append((eid, "", ",".join(all_cands_list)))
                continue
                
            scored = []
            for c in cands:
                sc = c['sc']
                f = c['f']
                if f[idx_s2] > 0.5: sc -= 0.09
                if f[idx_country_conf] > 0.5: sc -= 0.50
                if f[idx_postal_conf] > 0.5 and f[idx_exact_name] < 0.5: sc -= 0.30
                if f[idx_house_conf] > 0.5 and f[idx_exact_name] < 0.5: sc -= 0.20
                if f[idx_cat_E] > 0.5: sc -= 0.08
                if f[idx_addr_ratio] > 85 and f[idx_name_ratio] < 40: sc -= 0.20
                scored.append({'id': c['id'], 'sc': sc, 'f': f})
                
            scored.sort(key=lambda x: x['sc'], reverse=True)
            top = scored[0]
            
            if top['sc'] < 0.73:
                accepted_tids = []
            else:
                accepted_tids = [top['id']]
                top_sc = top['sc']
                for cand in scored[1:]:
                    c_sc = cand['sc']
                    gap = top_sc - c_sc
                    f = cand['f']
                    if c_sc >= 0.35 and gap <= 0.10 and f[idx_country_conf] < 0.5 and f[idx_postal_conf] < 0.5:
                        accepted_tids.append(cand['id'])
                        if len(accepted_tids) >= 8:
                            break
                    elif gap > 0.18:
                        break
                        
            # MANDATORY VERIFICATION: Candidate containment
            for acc in accepted_tids:
                if acc not in all_cands_list:
                    all_cands_list.append(acc)
                    
            matched_str = ",".join(accepted_tids)
            cands_str = ",".join(all_cands_list)
            batch_db_rows.append((eid, matched_str, cands_str))
            
        # Step D: Checkpoint to SQLite DB
        fresh_conn.executemany("INSERT OR REPLACE INTO preds VALUES (?, ?, ?);", batch_db_rows)
        fresh_conn.commit()
        
        n_processed += len(batch_eids)
        now = time.time()
        elapsed = now - t_query_start
        rate = n_processed / elapsed if elapsed > 0 else 0
        rem_sec = (total_to_process - n_processed) / rate if rate > 0 else 0
        
        # Periodic 300-second status report or milestone report
        if (now - last_report_time >= 300.0) or (n_processed % 30000 == 0) or (n_processed == total_to_process):
            last_report_time = now
            mem = psutil.Process().memory_info().rss / (1024**2)
            cpu = psutil.cpu_percent()
            vram_mb = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"  [{country}] Status Report | Processed: {n_processed:,}/{total_to_process:,} ({n_processed/total_to_process*100:.1f}%) | Speed: {rate:.1f} queries/sec | RAM: {mem:.1f}MB | CPU: {cpu}% | VRAM: {vram_mb:.1f}MB | ETA: {rem_sec/60:.1f} min", flush=True)

    fresh_conn.close()
    
    # Cleanup memory
    del target_dict, s1_dict, s1_cands_map, all_needed_target_ids, name_freq_dict, addr_freq_dict, token_df, token_idf
    gc.collect()
    print(f"Country {country} fully completed and checkpointed in {time.time() - t_stage_start:.1f}s.", flush=True)

def assemble_fresh_tsvs():
    print("\n" + "=" * 60, flush=True)
    print("ASSEMBLING FRESH PRODUCTION TSVS IN EXACT TEST S1 SEQUENCE", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    
    conn = sqlite3.connect(DB_FRESH, timeout=120.0)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM preds;")
    cnt = cur.fetchone()[0]
    print(f"Total rows in fresh DB: {cnt:,} (Required: {TOTAL_REQUIRED_S1:,})", flush=True)
    assert cnt == TOTAL_REQUIRED_S1, f"ERROR: fresh DB row count {cnt} != required {TOTAL_REQUIRED_S1}"
    
    f_match = open(OUT_MATCHING, "w", encoding="utf-8", newline="\n")
    f_cand = open(OUT_CANDIDATE, "w", encoding="utf-8", newline="\n")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    BATCH_SIZE = 50000
    current_ids = []
    
    total_written = 0
    empty_s1 = 0
    non_empty_s1 = 0
    total_links = 0
    s2_links = 0
    s3_links = 0
    total_cands = 0
    pred_counts = []
    
    def flush_batch(ids):
        nonlocal total_written, empty_s1, non_empty_s1, total_links, s2_links, s3_links, total_cands
        placeholders = ",".join("?" for _ in ids)
        cur.execute(f"SELECT s1_id, matched, candidates FROM preds WHERE s1_id IN ({placeholders});", ids)
        res_map = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        
        for eid in ids:
            matched, cand = res_map.get(eid, ("", ""))
            
            f_match.write(f"{eid}\t{matched}\n")
            f_cand.write(f"{eid}\t{cand}\n")
            total_written += 1
            
            m_list = [x for x in matched.split(",") if x.strip()]
            c_list = [x for x in cand.split(",") if x.strip()]
            
            # Max K <= 8 safety check
            assert len(m_list) <= 8, f"K-cap error for {eid}: {len(m_list)} > 8"
            
            # 100% Candidate Containment check
            for m in m_list:
                assert m in c_list, f"CRITICAL CONTAINMENT ERROR: {m} not in candidates for {eid}!"
                
            if m_list:
                non_empty_s1 += 1
                total_links += len(m_list)
                for m in m_list:
                    if m.startswith("S2-"): s2_links += 1
                    elif m.startswith("S3-"): s3_links += 1
            else:
                empty_s1 += 1
                
            pred_counts.append(len(m_list))
            total_cands += len(c_list)
            
    with open(TEST_S1_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r and len(r) >= 1:
                current_ids.append(r[0].strip())
                if len(current_ids) >= BATCH_SIZE:
                    flush_batch(current_ids)
                    current_ids = []
                    if total_written % 300000 == 0:
                        print(f"  Wrote {total_written:,}/{TOTAL_REQUIRED_S1:,} rows...", flush=True)
                        
    if current_ids:
        flush_batch(current_ids)
        
    f_match.flush()
    f_match.close()
    f_cand.flush()
    f_cand.close()
    conn.close()
    
    print(f"Assembly complete in {time.time() - t0:.1f}s.", flush=True)
    
    # Calculate SHA256
    m_sha = get_sha256(OUT_MATCHING)
    c_sha = get_sha256(OUT_CANDIDATE)
    
    print("\n" + "=" * 60, flush=True)
    print("FRESH V2 PRODUCTION INTEGRITY & SHA256 VERIFICATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Written Data Rows : {total_written:,} (Expected: {TOTAL_REQUIRED_S1:,})", flush=True)
    print(f"Non-empty S1 Entities   : {non_empty_s1:,} ({non_empty_s1/total_written*100:.2f}%)", flush=True)
    print(f"Empty S1 Entities       : {empty_s1:,} ({empty_s1/total_written*100:.2f}%)", flush=True)
    print(f"Total Matched Links     : {total_links:,} (S2: {s2_links:,}, S3: {s3_links:,})", flush=True)
    print(f"Mean Matches / Entity   : {float(np.mean(pred_counts)):.4f}", flush=True)
    print(f"Max Matches / Entity    : {int(np.max(pred_counts))}", flush=True)
    print(f"Candidate Containment   : 100.00% (0 violations)", flush=True)
    print(f"OLD Matching SHA256     : {OLD_MATCHING_SHA}", flush=True)
    print(f"FRESH Matching SHA256   : {m_sha}", flush=True)
    print(f"FRESH Candidate SHA256  : {c_sha}", flush=True)
    
    assert total_written == TOTAL_REQUIRED_S1, f"Row count error: {total_written} != {TOTAL_REQUIRED_S1}"
    assert m_sha != OLD_MATCHING_SHA, f"CRITICAL ERROR: Fresh matching SHA256 is identical to old SHA: {m_sha}!"
    print("\nVERIFIED: Fresh matching SHA256 is genuinely DIFFERENT from old SHA256!", flush=True)
    
    return {
        'total_written': total_written,
        'non_empty': non_empty_s1,
        'empty': empty_s1,
        'total_links': total_links,
        's2_links': s2_links,
        's3_links': s3_links,
        'm_sha': m_sha,
        'c_sha': c_sha
    }

def main():
    print("=" * 80, flush=True)
    print("STARTING FRESH V2 FULL-TEST PRODUCTION INFERENCE PIPELINE", flush=True)
    print("=" * 80, flush=True)
    t_global_start = time.time()
    
    # 1. Verify GPU
    verify_gpu_hardware()
    
    # 2. Load trained V2 model
    print(f"Loading validated LightGBM model from {MODEL_PATH}...", flush=True)
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"Model loaded: {model.num_trees()} trees, {model.num_feature()} features.\n", flush=True)
    
    # 3. Process countries in sequence: FRANCE -> US -> INDIA
    for ctry in ["FRANCE", "US", "INDIA"]:
        run_country_inference(ctry, model)
        
    # 4. Assemble fresh TSVs and verify
    stats = assemble_fresh_tsvs()
    
    print("\n" + "=" * 80, flush=True)
    print(f"FRESH V2 PRODUCTION INFERENCE FULLY COMPLETED in {time.time() - t_global_start:.1f}s", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
