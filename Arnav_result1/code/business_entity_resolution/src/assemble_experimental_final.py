import os
import sys
import csv
import time
import sqlite3
import numpy as np

DATA_DIR = r"D:\dataset\student_resource\dataset\test"
DB_PATH = r"D:\Arnav\cache\full_test_pipeline.db"
EXP_DIR = r"D:\Arnav\output\experimental_final"
os.makedirs(EXP_DIR, exist_ok=True)

OUT_MATCHING = os.path.join(EXP_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(EXP_DIR, "candidate_pairs.tsv")
TEST_S1 = os.path.join(DATA_DIR, "test_source1.tsv")

TOTAL_REQUIRED = 1732544

def main():
    print("=" * 60)
    print("ASSEMBLING EXPERIMENTAL FINAL PRODUCTION TSVS (V5+ PRECISION ENHANCED)")
    print("=" * 60)
    t0 = time.time()
    
    conn = sqlite3.connect(DB_PATH, timeout=120.0)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM preds;")
    cnt = cur.fetchone()[0]
    print(f"Total rows in preds DB: {cnt:,} (Required: {TOTAL_REQUIRED:,})")
    assert cnt == TOTAL_REQUIRED, f"Mismatch: {cnt} != {TOTAL_REQUIRED}"
    
    f_match = open(OUT_MATCHING, "w", encoding="utf-8", newline="")
    f_cand = open(OUT_CANDIDATE, "w", encoding="utf-8", newline="")
    
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
    
    def process_batch(ids):
        nonlocal total_written, empty_s1, non_empty_s1, total_links, s2_links, s3_links, total_cands
        res_map = {}
        SUB = 900
        for s in range(0, len(ids), SUB):
            sub_ids = ids[s : s + SUB]
            placeholders = ",".join("?" for _ in sub_ids)
            cur.execute(f"SELECT s1_id, matched, candidates FROM preds WHERE s1_id IN ({placeholders});", sub_ids)
            for r in cur.fetchall():
                res_map[r[0]] = (r[1], r[2])
        
        m_lines = []
        c_lines = []
        
        for eid in ids:
            m_str, c_str = res_map.get(eid, ("", ""))
            
            # Parse and deduplicate matches
            raw_m = [x.strip() for x in m_str.split(",") if x.strip()]
            seen_m = set()
            dedup_m = []
            for m in raw_m:
                if m not in seen_m:
                    seen_m.add(m)
                    dedup_m.append(m)
                    
            # Precision-protective Max K = 8 capping
            if len(dedup_m) > 8:
                dedup_m = dedup_m[:8]
                
            # Parse candidates
            raw_c = [x.strip() for x in c_str.split(",") if x.strip()]
            seen_c = set()
            dedup_c = []
            for c in raw_c:
                if c not in seen_c:
                    seen_c.add(c)
                    dedup_c.append(c)
                    
            # Enforce 100% Candidate Containment
            for m in dedup_m:
                if m not in seen_c:
                    seen_c.add(m)
                    dedup_c.append(m)
                    
            # Check statistics
            if dedup_m:
                non_empty_s1 += 1
                total_links += len(dedup_m)
                for m in dedup_m:
                    if m.startswith("S2-"): s2_links += 1
                    elif m.startswith("S3-"): s3_links += 1
            else:
                empty_s1 += 1
                
            pred_counts.append(len(dedup_m))
            total_cands += len(dedup_c)
            total_written += 1
            
            m_lines.append(f"{eid}\t{','.join(dedup_m)}\n")
            c_lines.append(f"{eid}\t{','.join(dedup_c)}\n")
            
        f_match.writelines(m_lines)
        f_cand.writelines(c_lines)
        
    print(f"Streaming from {TEST_S1} in exact sequence...")
    with open(TEST_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if r and len(r) >= 1:
                eid = r[0].strip()
                current_ids.append(eid)
                if len(current_ids) >= BATCH_SIZE:
                    process_batch(current_ids)
                    current_ids = []
                    if total_written % 200000 == 0:
                        print(f"  Processed {total_written:,}/{TOTAL_REQUIRED:,} rows ({total_written/TOTAL_REQUIRED*100:.1f}%)...")
                        
    if current_ids:
        process_batch(current_ids)
        
    f_match.flush()
    f_match.close()
    f_cand.flush()
    f_cand.close()
    conn.close()
    
    elapsed = time.time() - t0
    print(f"\nAssembly complete in {elapsed:.1f}s!")
    print(f"Total Written: {total_written:,}")
    print(f"Empty S1 (Singletons): {empty_s1:,} ({empty_s1/total_written*100:.2f}%)")
    print(f"Non-empty S1: {non_empty_s1:,} ({non_empty_s1/total_written*100:.2f}%)")
    print(f"Total Predicted Links: {total_links:,}")
    print(f"  S2 Links: {s2_links:,} ({s2_links/total_links*100:.1f}%)")
    print(f"  S3 Links: {s3_links:,} ({s3_links/total_links*100:.1f}%)")
    print(f"Mean Matches/Entity: {np.mean(pred_counts):.4f}")
    print(f"Median Matches: {np.percentile(pred_counts, 50):.1f}")
    print(f"P90 Matches: {np.percentile(pred_counts, 90):.1f}")
    print(f"P95 Matches: {np.percentile(pred_counts, 95):.1f}")
    print(f"P99 Matches: {np.percentile(pred_counts, 99):.1f}")
    print(f"Max Matches: {np.max(pred_counts)}")
    print(f"Total Candidate Pairs: {total_cands:,} (Avg {total_cands/total_written:.2f}/entity)")
    
    # Save Assembly Stats
    with open(os.path.join(EXP_DIR, "assembly_stats.txt"), "w", encoding="utf-8") as out_f:
        out_f.write(f"total_written={total_written}\n")
        out_f.write(f"empty_s1={empty_s1}\n")
        out_f.write(f"non_empty_s1={non_empty_s1}\n")
        out_f.write(f"total_links={total_links}\n")
        out_f.write(f"s2_links={s2_links}\n")
        out_f.write(f"s3_links={s3_links}\n")
        out_f.write(f"mean_matches={np.mean(pred_counts):.4f}\n")
        out_f.write(f"max_matches={np.max(pred_counts)}\n")
        out_f.write(f"total_cands={total_cands}\n")

if __name__ == "__main__":
    main()
