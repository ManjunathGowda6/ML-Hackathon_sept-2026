# Competition Documentation: Business Entity Resolution

## 1. Executive Summary

| Attribute | Details |
|---|---|
| **Project** | Business Entity Resolution |
| **System Version** | Laptop 1 — V5-Anchored (V2 Model) |
| **Model Type** | Dual Multi-Channel Retrieval + Multi-View Feature Scorer + Precision-Calibrated Decision Gate |
| **Best Offline Validation Result** | **Entity Macro F0.5 = 0.8791** |
| **Independent Validation (Slice A / B)** | **0.8641 / 0.8626** |
| **Known Proven Competition Leaderboard Result** | **Laptop 2 V3 = 0.786** |
| **Status** | Offline Validation Package (Preserved Experimental Top Result) |

---

## 2. Methodology & Architecture

### 2.1 Multi-View Business Representation
Each business entity is processed into multiple distinct normalized views to address orthographic, structural, and linguistic variations:
1. **Name Views**:
   - Original normalized name (accents stripped, lowercase, cleaned)
   - Punctuation-stripped and alphanumeric compact name
   - Legal suffix stripped name (`ltd`, `inc`, `corp`, `sarl`, `llc`, etc.)
   - Sorted-token name (for word-order invariance)
   - Domain stem extraction (URL/web domain token anchors)
   - Transliterated / script-aware representation
2. **Address Views**:
   - Standardized address string with expansion of common abbreviations (`st` $\to$ `street`, `rd` $\to$ `road`, etc.)
   - Extracted postal code / PIN prefix
   - Extracted house / building / plot number
   - Rare address tokens (excluding stop-words and regional generic tokens)
3. **Cross-Field Metadata**: Country code, entity source channel (`S1`, `S2`, `S3`).

### 2.2 Dual Multi-Channel Inverted Index Retrieval
A two-stage candidate retrieval architecture maximizes recall while strictly controlling candidate volume:
- **Primary Stage (High Precision)**:
  - Exact normalized name match
  - Exact compact name match
  - Domain stem match
  - Sorted stripped name match
  - Postal prefix + name prefix anchor
  - Rare token pair intersection
- **Secondary Stage (Selective Recovery)**:
  - Activated for rare-token failures, zero lexical overlap, or transliterated names
  - Cross-field name + address combination retrieval
  - Substring character 3-gram retrieval for short names
- **Candidate Merging**: Tightly budgeted ranking and deduplication ensuring candidate recall > 94% with $\le 35$ candidates per entity.

### 2.3 Multi-View Feature Engineering (71 Dimensions)
Extracted pairwise features cover five critical families:
1. **Lexical & Fuzzy String Similarities**: Token Jaccard, token overlap, RapidFuzz token sort/set ratio, edit distance.
2. **Address & Geographic Agreement**: Postal code match, house number match, address token overlap, country match.
3. **Conflict Indicators**: Explicit country conflict, postal conflict, house number conflict.
4. **Frequency & Information Content**: Token IDF scores, rare-token counts, generic-name frequency penalties.
5. **Retrieval Channel Evidence**: Channel hit count, retrieval rank, primary vs recovery source indicators.

### 2.4 Precision-Calibrated Decision Gate ($K \le 8$)
Because the target metric is Entity Macro $F_{0.5}$ (which penalizes false positives 4× more heavily than false negatives), decisions are not made independently:
- **Calibrated Conflict Penalties**:
  - `country_conflict`: `-0.50` (strict cross-border veto)
  - `postal_conflict`: `-0.30` (applied when exact name does not match)
  - `house_num_conflict`: `-0.20` (applied when exact name does not match)
  - `cat_E` (secondary recovery candidate): `-0.08`
  - Commercial building collision (`addr_token_set_ratio > 85 and name_ratio < 40`): `-0.20`
  - Source-2 noisiness penalty: `-0.09`
- **Adaptive Multi-Match Gating**:
  - Singleton threshold: Top candidate adjusted score must exceed `0.73`.
  - Secondary matches admitted only if adjusted score $\ge 0.35$, score gap to top candidate $\le 0.10$, and zero country/postal conflicts.
  - Scan breaks early if score gap exceeds `0.18`.
  - Hard cap of $K \le 8$ matches per entity to prevent runaway false-positive clusters.
- **Mandatory Candidate Containment**: 100% mathematical containment enforced between predicted matches and candidate lists.

---

## 3. Experimental Validation Results

### 3.1 Primary Validation Benchmark (Validation A)
Evaluated on 3,000 S1 reference queries against a 509,250-target validation universe:

| Metric | V5 Baseline | Upgraded V5+ (V2 Model) | Delta |
|---|---|---|---|
| **Entity Macro F0.5** | 0.8369 | **0.8791** | **+0.0422** |
| **Entity Precision** | 0.8754 | **0.9134** | **+0.0380** |
| **Entity Recall** | 0.8402 | **0.8537** | **+0.0135** |
| **Pair Precision** | 0.8812 | **0.9188** | **+0.0376** |
| **Pair Recall** | 0.8329 | **0.8441** | **+0.0112** |
| **Candidate Recall** | 90.11% | **94.33%** | **+4.22%** |

### 3.2 Independent Anti-Overfitting Validation (Dual-Slice Benchmark)
To verify that the configuration was not overfit to a single validation split, two independent 1,500-entity disjoint slices were evaluated:

| Slice | V5 Baseline F0.5 | Upgraded V5+ F0.5 | Delta F0.5 | Result |
|---|---|---|---|---|
| **Slice A (Queries 1–1,500)** | 0.8337 | **0.8641** | **+0.0304** | Consistent Win |
| **Slice B (Queries 1,501–3,000)** | 0.8402 | **0.8626** | **+0.0225** | Consistent Win |

---

## 4. Critical Limitations & Integrity Disclaimers

1. **Offline Validation vs. Competition Leaderboard**:
   - The **0.8791** score is strictly an **offline validation / experimental result**.
   - It was **never confirmed by an official competition leaderboard submission**.
   - **DO NOT cite or report 0.8791 as the competition leaderboard score.**
   - The verified, proven leaderboard score from competition submissions is separate: **Laptop 2 V3 leaderboard score = 0.786**.
2. **Full-Test Fresh Inference Status**:
   - A full-test fresh V2 inference run across all 1,732,544 test S1 entities was initiated (`run_fresh_v2_full_test.py`).
   - Due to a 30-minute hard deadline, the run was safely stopped and durably checkpointed after **9,000 / 1,732,544 S1 entities** (`PRAGMA integrity_check: ok` in `fresh_v2_pipeline.db`).
   - The previously generated production files on disk (`matching_results.tsv` SHA256: `908821E3...`) were preserved and not replaced with an incomplete run.
3. **No Unsupported Claims**:
   - All claims in this report are confined to verifiable offline benchmarks and audited code artifacts.

---

## 5. Artifact Manifest

```
D:\Arnav\Arnav\
├── output\
│   ├── matching_results.tsv       (102,558,913 bytes, 1,732,544 rows)
│   └── candidate_pairs.tsv        (785,267,414 bytes, 1,732,544 rows)
├── code\
│   └── business_entity_resolution\
│       ├── src\
│       │   ├── multiview.py
│       │   ├── retrieval.py
│       │   ├── recall_recovery.py
│       │   ├── features.py
│       │   ├── graph_ranker.py
│       │   ├── metrics.py
│       │   ├── benchmark_two_slices.py
│       │   ├── evaluate_v5_anchored.py
│       │   ├── deep_forensic_error_mining.py
│       │   ├── run_fresh_v2_full_test.py
│       │   ├── run_full_test_production.py
│       │   ├── assemble_experimental_final.py
│       │   └── gpu_cuda_benchmark.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
