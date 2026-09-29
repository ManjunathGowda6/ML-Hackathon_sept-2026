import os
import sys
sys.path.insert(0, r"D:\Arnav")
sys.stdout.reconfigure(line_buffering=True)
import csv
import time
import gc
import sqlite3
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter

from scripts.multiview import (
    MultiViewRecord, normalize_text, compact_text, strip_legal_suffixes,
    extract_domain_stem, extract_postal_code, extract_house_number,
    sorted_tokens, normalize_address, GENERIC_ADDR_TOKENS
)
from scripts.features import extract_upgraded_pair_vector, UPGRADED_FEATURE_COLUMNS
from scripts.graph_ranker import EntityMatchDecider

DATA_DIR = r"D:\dataset\student_resource\dataset\test"
OUTPUT_DIR = r"D:\Arnav\output"
CACHE_DIR = r"D:\Arnav\cache"
MODELS_DIR = r"D:\Arnav\models"
REPORTS_DIR = r"D:\Arnav\reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DB_PATH = os.path.join(CACHE_DIR, "full_test_pipeline.db")
MODEL_PATH = os.path.join(MODELS_DIR, "pair_match_model_v2.txt")
TEST_S1_FILE = os.path.join(DATA_DIR, "test_source1.tsv")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

TOTAL_REQUIRED_S1 = 1732544

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=120.0)
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

def extract_target_index_keys(bname: str, baddr: str, ctry: str):
    """
    Lightweight signature extractor for fast inverted index indexing.
    """
    norm_n = normalize_text(bname)
    comp_n = compact_text(norm_n)
    stem_n = extract_domain_stem(bname)
    strip_n = strip_legal_suffixes(norm_n)
    comp_strip_n = compact_text(strip_n)
    sort_strip_n = " ".join(sorted(strip_n.split())) if strip_n else ""
    
    toks_n = [t for t in norm_n.split() if len(t) >= 2]
    if stem_n and stem_n not in toks_n:
        toks_n.append(stem_n)
        
    norm_a = normalize_address(baddr)
    toks_a = [t for t in norm_a.split() if len(t) >= 4 and t not in GENERIC_ADDR_TOKENS and not t.isdigit()]
    postal = extract_postal_code(baddr, ctry)
    house = extract_house_number(norm_a)
    
    return {
        'norm_n': norm_n,
        'comp_n': comp_n,
        'stem_n': stem_n,
        'strip_n': strip_n,
        'comp_strip_n': comp_strip_n,
        'sort_strip_n': sort_strip_n,
        'toks_n': toks_n,
        'toks_a': toks_a,
        'postal': postal,
        'house': house
    }

def run_country_stage(country: str, model: lgb.Booster, decider: EntityMatchDecider):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE: Processing Country: {country}", flush=True)
    print("=" * 60, flush=True)
    t_stage_start = time.time()
    
    conn = init_db()
    
    # 1. Identify remaining S1 entities for this country
    print(f"Scanning {TEST_S1_FILE} for {country} entities...", flush=True)
    s1_items = []
    with open(TEST_S1_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r and len(r) >= 4 and r[3].strip().upper() == country:
                s1_items.append((r[0].strip(), r[1], r[2], r[3]))
                
    total_country_s1 = len(s1_items)
    print(f"Found {total_country_s1:,} S1 entities for {country}.", flush=True)
    
    # Check already done
    cur = conn.cursor()
    cur.execute("SELECT s1_id FROM preds;")
    done_ids = {row[0] for row in cur.fetchall()}
    remaining_s1 = [it for it in s1_items if it[0] not in done_ids]
    del s1_items
    gc.collect()
    
    if not remaining_s1:
        print(f"Country {country} already 100% complete! Moving to next stage.", flush=True)
        conn.close()
        return
        
    print(f"Remaining S1 entities to process for {country}: {len(remaining_s1):,}.", flush=True)
    
    # 2. Ingest Target Records into SQLite and build Lightweight Inverted Indices
    conn.execute("DROP TABLE IF EXISTS targets;")
    conn.execute("CREATE TABLE targets (target_id TEXT PRIMARY KEY, bname TEXT, baddr TEXT, ctry TEXT);")
    conn.commit()
    
    print(f"Streaming target catalog for {country} into SQLite and building inverted indices...", flush=True)
    t_idx_start = time.time()
    
    # Inverted Index Postings: Key -> list of target_id
    idx_exact_name = defaultdict(list)
    idx_exact_compact = defaultdict(list)
    idx_domain_stem = defaultdict(list)
    idx_sorted_name = defaultdict(list)
    idx_name_token = defaultdict(list)
    idx_name_addr_comb = defaultdict(list)
    idx_addr_house_street = defaultdict(list)
    idx_addr_postal_street = defaultdict(list)
    idx_postal_prefix = defaultdict(list)
    idx_rare_token_pair = defaultdict(list)
    idx_char_3gram = defaultdict(list)
    
    token_df = Counter()
    addr_df = Counter()
    name_freq_dict = Counter()
    addr_freq_dict = Counter()
    
    target_count = 0
    batch_targets = []
    
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join(DATA_DIR, fname)
        t_src_start = time.time()
        c_src = 0
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for r in reader:
                if r and len(r) >= 4 and r[3].strip().upper() == country:
                    tid, bname, baddr, ctry = r[0].strip(), r[1], r[2], r[3]
                    batch_targets.append((tid, bname, baddr, ctry))
                    c_src += 1
                    target_count += 1
                    
                    # Extract keys
                    keys = extract_target_index_keys(bname, baddr, ctry)
                    
                    if keys['norm_n']:
                        name_freq_dict[keys['norm_n']] += 1
                        if len(idx_exact_name[keys['norm_n']]) < 50:
                            idx_exact_name[keys['norm_n']].append(tid)
                            
                    if keys['comp_strip_n'] and len(keys['comp_strip_n']) >= 3:
                        if len(idx_exact_compact[keys['comp_strip_n']]) < 50:
                            idx_exact_compact[keys['comp_strip_n']].append(tid)
                            
                    if keys['stem_n'] and len(keys['stem_n']) >= 3:
                        if len(idx_domain_stem[keys['stem_n']]) < 50:
                            idx_domain_stem[keys['stem_n']].append(tid)
                            
                    if keys['sort_strip_n'] and len(keys['sort_strip_n']) >= 3:
                        if len(idx_sorted_name[keys['sort_strip_n']]) < 50:
                            idx_sorted_name[keys['sort_strip_n']].append(tid)
                            
                    # Name tokens (count DF)
                    for t in keys['toks_n']:
                        token_df[t] += 1
                        if len(idx_name_token[t]) < 50:
                            idx_name_token[t].append(tid)
                            
                    # Address tokens
                    if keys['toks_a']:
                        addr_freq_dict[keys['norm_n']] += 1
                        for at in keys['toks_a'][:2]:
                            addr_df[at] += 1
                            
                    # House + Street
                    if keys['house'] and keys['toks_a']:
                        h_key = (keys['house'], keys['toks_a'][0])
                        if len(idx_addr_house_street[h_key]) < 40:
                            idx_addr_house_street[h_key].append(tid)
                            
                    # Postal + Street
                    if keys['postal'] and keys['toks_a']:
                        p_key = (keys['postal'], keys['toks_a'][0])
                        if len(idx_addr_postal_street[p_key]) < 40:
                            idx_addr_postal_street[p_key].append(tid)
                            
                    # Postal + Prefix
                    if keys['postal'] and keys['norm_n']:
                        pref_key = (keys['postal'], keys['norm_n'][:3])
                        if len(idx_postal_prefix[pref_key]) < 40:
                            idx_postal_prefix[pref_key].append(tid)
                            
                    # Rare token pair
                    if len(keys['toks_n']) >= 2:
                        t1, t2 = keys['toks_n'][0], keys['toks_n'][1]
                        pair_k = (min(t1, t2), max(t1, t2))
                        if len(idx_rare_token_pair[pair_k]) < 40:
                            idx_rare_token_pair[pair_k].append(tid)
                            
                    # Name + Addr comb
                    if keys['toks_n'] and keys['toks_a']:
                        comb_k = (keys['toks_n'][0], keys['toks_a'][0])
                        if len(idx_name_addr_comb[comb_k]) < 40:
                            idx_name_addr_comb[comb_k].append(tid)
                            
                    # Char 3-grams (only for short names <= 15 chars)
                    c_name = keys['comp_strip_n']
                    if 3 <= len(c_name) <= 15:
                        for i in range(min(4, len(c_name) - 2)):
                            ng = c_name[i:i+3]
                            if len(idx_char_3gram[ng]) < 30:
                                idx_char_3gram[ng].append(tid)
                                
                    if len(batch_targets) >= 50000:
                        conn.executemany("INSERT INTO targets VALUES (?, ?, ?, ?);", batch_targets)
                        conn.commit()
                        batch_targets = []
                        
        if batch_targets:
            conn.executemany("INSERT INTO targets VALUES (?, ?, ?, ?);", batch_targets)
            conn.commit()
            batch_targets = []
            
        print(f"  Streamed {c_src:,} records from {fname} in {time.time() - t_src_start:.1f}s.", flush=True)
        
    print(f"Total {country} targets indexed: {target_count:,} in {time.time() - t_idx_start:.1f}s.", flush=True)
    
    # Compute IDF dictionary
    N = float(target_count)
    token_idf = {t: np.log((N + 1.0) / (df + 1.0)) + 1.0 for t, df in token_df.items()}
    
    # 3. Process Queries in Batches
    BATCH_SIZE = 2500
    n_processed = 0
    total_to_process = len(remaining_s1)
    t_query_start = time.time()
    
    print(f"Processing {total_to_process:,} {country} S1 entities in batches of {BATCH_SIZE}...", flush=True)
    
    for b_start in range(0, total_to_process, BATCH_SIZE):
        batch_items = remaining_s1[b_start : b_start + BATCH_SIZE]
        batch_t0 = time.time()
        
        # S1 candidates map: s1_id -> candidate_tids (ordered list)
        s1_candidates_map = {}
        s1_records_map = {}
        all_candidate_tids = set()
        
        # Step 1: Candidate Retrieval
        for eid, bname, baddr, ctry in batch_items:
            r1 = MultiViewRecord(eid, bname, baddr, ctry)
            s1_records_map[eid] = r1
            
            cands = Counter()
            
            # Exact Name / Compact / Stem / Sorted
            if r1.norm_name in idx_exact_name:
                for tid in idx_exact_name[r1.norm_name]: cands[tid] += 5.0
            if r1.compact_stripped_name in idx_exact_compact:
                for tid in idx_exact_compact[r1.compact_stripped_name]: cands[tid] += 4.5
            if r1.domain_stem and r1.domain_stem in idx_domain_stem:
                for tid in idx_domain_stem[r1.domain_stem]: cands[tid] += 5.0
            if r1.legal_stripped_name:
                sort_s1 = " ".join(sorted(r1.legal_stripped_name.split()))
                if sort_s1 in idx_sorted_name:
                    for tid in idx_sorted_name[sort_s1]: cands[tid] += 4.0
                    
            # Postal Prefix & Address Anchors
            if r1.postal_code and r1.norm_name:
                pref_k = (r1.postal_code, r1.norm_name[:3])
                if pref_k in idx_postal_prefix:
                    for tid in idx_postal_prefix[pref_k]: cands[tid] += 3.5
                    
            if r1.house_number and r1.rare_tokens_address:
                h_k = (r1.house_number, r1.rare_tokens_address[0])
                if h_k in idx_addr_house_street:
                    for tid in idx_addr_house_street[h_k]: cands[tid] += 3.5
                    
            if r1.postal_code and r1.rare_tokens_address:
                p_k = (r1.postal_code, r1.rare_tokens_address[0])
                if p_k in idx_addr_postal_street:
                    for tid in idx_addr_postal_street[p_k]: cands[tid] += 3.0
                    
            # Rare token pair
            if len(r1.tokens_name) >= 2:
                # Pick 2 rarest tokens by IDF
                s_toks = sorted(r1.tokens_name, key=lambda t: token_idf.get(t, 1.0), reverse=True)
                pair_k = (min(s_toks[0], s_toks[1]), max(s_toks[0], s_toks[1]))
                if pair_k in idx_rare_token_pair:
                    for tid in idx_rare_token_pair[pair_k]: cands[tid] += 3.5
                    
            # Combined name + addr
            if r1.tokens_name and r1.rare_tokens_address:
                comb_k = (r1.tokens_name[0], r1.rare_tokens_address[0])
                if comb_k in idx_name_addr_comb:
                    for tid in idx_name_addr_comb[comb_k]: cands[tid] += 3.5
                    
            # Rare Name Tokens
            for t in r1.tokens_name[:3]:
                if t in idx_name_token and token_df.get(t, 0) <= 8000:
                    idf = token_idf.get(t, 2.0)
                    for tid in idx_name_token[t][:30]:
                        cands[tid] += min(3.0, idf * 0.5)
                        
            # Char 3-grams
            c_name = r1.compact_stripped_name
            if c_name and len(c_name) <= 15:
                for i in range(min(4, len(c_name) - 2)):
                    ng = c_name[i:i+3]
                    if ng in idx_char_3gram:
                        for tid in idx_char_3gram[ng][:25]:
                            cands[tid] += 1.0
                            
            # Prune to top 35 candidates
            top_cands = [tid for tid, _ in cands.most_common(35)]
            s1_candidates_map[eid] = top_cands
            all_candidate_tids.update(top_cands)
            
        # Step 2: Fetch Target Records from SQLite in Bulk
        cur = conn.cursor()
        target_objs_map = {}
        if all_candidate_tids:
            cand_list = list(all_candidate_tids)
            SUB_CHUNK = 25000
            for sc_start in range(0, len(cand_list), SUB_CHUNK):
                sub_ids = cand_list[sc_start : sc_start + SUB_CHUNK]
                placeholders = ",".join("?" for _ in sub_ids)
                cur.execute(f"SELECT target_id, bname, baddr, ctry FROM targets WHERE target_id IN ({placeholders});", sub_ids)
                for tid, bn, ba, ct in cur.fetchall():
                    target_objs_map[tid] = MultiViewRecord(tid, bn, ba, ct)
                    
        # Step 3: Feature Extraction
        feature_rows = []
        pair_meta = []
        
        for eid, cand_tids in s1_candidates_map.items():
            r1 = s1_records_map[eid]
            for rank, tid in enumerate(cand_tids, 1):
                if tid in target_objs_map:
                    r2 = target_objs_map[tid]
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
                    f_vec = extract_upgraded_pair_vector(r1, r2, ret_info, name_freq_dict, addr_freq_dict, token_idf)
                    feature_rows.append(f_vec)
                    pair_meta.append((eid, tid, f_vec))
                    
        # Step 4: LightGBM Batch Scoring
        if feature_rows:
            X = np.array(feature_rows, dtype=np.float32)
            raw_scores = model.predict(X, num_threads=10)
        else:
            raw_scores = []
            
        cand_by_s1 = defaultdict(list)
        for i, sc in enumerate(raw_scores):
            eid, tid, f_vec = pair_meta[i]
            cand_by_s1[eid].append({
                'target_id': tid,
                'raw_score': float(sc),
                'is_source_2': 1.0 if tid.startswith('S2-') else 0.0,
                'postal_conflict': f_vec[UPGRADED_FEATURE_COLUMNS.index('postal_conflict')],
                'country_conflict': f_vec[UPGRADED_FEATURE_COLUMNS.index('country_conflict')],
                'house_num_conflict': f_vec[UPGRADED_FEATURE_COLUMNS.index('house_num_conflict')],
                'name_exact_match': f_vec[UPGRADED_FEATURE_COLUMNS.index('name_exact_match')],
                'channel_hit_count': f_vec[UPGRADED_FEATURE_COLUMNS.index('channel_hit_count')],
            })
            
        # Step 5: Entity-Level Decision & Mandatory Containment
        batch_db_rows = []
        for eid, _, _, _ in batch_items:
            cands_for_entity = cand_by_s1.get(eid, [])
            accepted_tids = decider.decide_matches_for_entity(cands_for_entity)
            all_cands_list = list(s1_candidates_map.get(eid, []))
            
            # MANDATORY VERIFICATION: Candidate containment
            for acc in accepted_tids:
                if acc not in all_cands_list:
                    all_cands_list.append(acc)
                    
            matched_str = ",".join(accepted_tids)
            cands_str = ",".join(all_cands_list)
            batch_db_rows.append((eid, matched_str, cands_str))
            
        # Step 6: Insert into SQLite
        conn.executemany("INSERT OR REPLACE INTO preds VALUES (?, ?, ?);", batch_db_rows)
        conn.commit()
        
        n_processed += len(batch_items)
        elapsed = time.time() - t_query_start
        rate = n_processed / elapsed if elapsed > 0 else 0
        rem_sec = (total_to_process - n_processed) / rate if rate > 0 else 0
        
        if n_processed % 10000 == 0 or n_processed == total_to_process:
            print(f"  [{country}] Progress: {n_processed:,}/{total_to_process:,} ({n_processed/total_to_process*100:.1f}%) | Speed: {rate:.1f} queries/sec | ETA: {rem_sec/60:.1f} min", flush=True)

    conn.close()
    
    # Cleanup memory for next country
    del idx_exact_name, idx_exact_compact, idx_domain_stem, idx_sorted_name, idx_name_token
    del idx_name_addr_comb, idx_addr_house_street, idx_addr_postal_street, idx_postal_prefix
    del idx_rare_token_pair, idx_char_3gram, token_df, addr_df, name_freq_dict, addr_freq_dict, token_idf
    gc.collect()
    
    print(f"Country {country} fully completed in {time.time() - t_stage_start:.1f}s.", flush=True)

def assemble_production_tsvs():
    print("\n" + "=" * 60, flush=True)
    print("ASSEMBLING FINAL PRODUCTION TSV FILES IN EXACT TEST S1 SEQUENCE", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    
    conn = init_db()
    cur = conn.cursor()
    
    cur.execute("SELECT count(*) FROM preds;")
    db_count = cur.fetchone()[0]
    print(f"Total predictions stored in SQLite: {db_count:,} (Required: {TOTAL_REQUIRED_S1:,}).", flush=True)
    assert db_count == TOTAL_REQUIRED_S1, f"ERROR: DB count {db_count} != required {TOTAL_REQUIRED_S1}"
    
    print(f"Writing {MATCHING_FILE} and {CANDIDATE_FILE}...", flush=True)
    
    f_match = open(MATCHING_FILE, "w", encoding="utf-8", newline="\n")
    f_cand = open(CANDIDATE_FILE, "w", encoding="utf-8", newline="\n")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    BATCH_SIZE = 30000
    current_ids = []
    
    total_written = 0
    empty_s1 = 0
    non_empty_s1 = 0
    total_matched_links = 0
    total_candidate_pairs = 0
    s2_links = 0
    s3_links = 0
    pred_counts = []
    
    def flush_batch(ids):
        nonlocal total_written, empty_s1, non_empty_s1, total_matched_links, total_candidate_pairs, s2_links, s3_links
        placeholders = ",".join("?" for _ in ids)
        cur.execute(f"SELECT s1_id, matched, candidates FROM preds WHERE s1_id IN ({placeholders});", ids)
        rows_dict = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        
        for eid in ids:
            if eid in rows_dict:
                matched, cand = rows_dict[eid]
            else:
                matched, cand = "", ""
                
            f_match.write(f"{eid}\t{matched}\n")
            f_cand.write(f"{eid}\t{cand}\n")
            total_written += 1
            
            m_list = [x for x in matched.split(",") if x.strip()]
            c_list = [x for x in cand.split(",") if x.strip()]
            
            if m_list:
                non_empty_s1 += 1
                total_matched_links += len(m_list)
                for m in m_list:
                    if m.startswith("S2-"): s2_links += 1
                    elif m.startswith("S3-"): s3_links += 1
            else:
                empty_s1 += 1
                
            pred_counts.append(len(m_list))
            total_candidate_pairs += len(c_list)
            
            # Strict candidate containment check
            for m in m_list:
                assert m in c_list, f"CRITICAL CONTAINMENT ERROR: {m} not in candidate_pairs for {eid}!"

    with open(TEST_S1_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r and len(r) >= 1:
                eid = r[0].strip()
                current_ids.append(eid)
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
    
    stats = {
        'total_written': total_written,
        'empty_s1': empty_s1,
        'non_empty_s1': non_empty_s1,
        'total_matched_links': total_matched_links,
        's2_links': s2_links,
        's3_links': s3_links,
        'mean_matches': float(np.mean(pred_counts)),
        'median_matches': float(np.percentile(pred_counts, 50)),
        'p90': float(np.percentile(pred_counts, 90)),
        'p95': float(np.percentile(pred_counts, 95)),
        'p99': float(np.percentile(pred_counts, 99)),
        'max_matches': int(np.max(pred_counts)),
        'total_candidate_pairs': total_candidate_pairs,
        'avg_candidate_pairs': total_candidate_pairs / total_written
    }
    
    print("\nFINAL OUTPUT DATASET STATISTICS:", flush=True)
    for k, v in stats.items():
        print(f"  {k:25s}: {v}", flush=True)
        
    return stats

def main():
    print("=" * 60, flush=True)
    print("STARTING FULL TEST PRODUCTION PIPELINE (RECALL-RECOVERY SYSTEM)", flush=True)
    print("=" * 60, flush=True)
    t_start = time.time()
    
    print(f"Loading trained LightGBM model from {MODEL_PATH}...", flush=True)
    model = lgb.Booster(model_file=MODEL_PATH)
    
    decider = EntityMatchDecider(
        min_top_score=0.73,
        multi_match_threshold=0.35,
        max_gap_from_top=0.10,
        s2_penalty=0.09
    )
    
    # Process sequentially: FRANCE -> US -> INDIA
    for ctry in ["FRANCE", "US", "INDIA"]:
        run_country_stage(ctry, model, decider)
        
    # Assemble final TSV files
    stats = assemble_production_tsvs()
    print(f"\nFull Test Pipeline successfully completed in {time.time() - t_start:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
