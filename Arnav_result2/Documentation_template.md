# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Arnav  
**Team Members:** Arnav  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We developed a multi-channel evidence candidate generation pipeline combined with a supervised pairwise ranking classifier (HistGradientBoostingClassifier with 32 discriminative features) and an adaptive, evidence-driven entity decision engine. The key innovation is a **V3 Multi-Channel Evidence Candidate Generator** that uses rarest-token prioritization, word-order-invariant sorted-token slugs, and address joint keys (house number + street word) to achieve 79% candidate recall against a 9.97M-entity test catalog. Combined with stratified hard-negative mining across four collision types, this yields an **official leaderboard score of 0.786**.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights discovered during EDA and forensic testing:

- **Distractor Flooding at Scale**: Validated systems that scored well against a small distractor set (~500K catalog) collapsed to leaderboard scores of 0.330–0.539 when evaluated against the full 9.97M test catalog, due to generic business tokens (`restaurant`, `hotel`, `consulting`, `services`, `LLC`) matching thousands of unrelated businesses.
- **Candidate Recall Bottleneck**: Rigid posting-list truncation caps caused 50.33% candidate recall on the full catalog in the baseline, meaning nearly half of all true matches were never even considered by the classifier.
- **Address Noise Patterns**: ~15% of Source 2 records had missing or empty addresses. Source 3 records frequently contained transliterated names (e.g., Indian/French business names romanized inconsistently).
- **Word-Order Variance**: Business names like "Hotel Royal" vs "Royal Hotel" represent the same entity; naive slug matching misses these.
- **Multi-Match Entities**: ~94% of S1 entities have at least one true match; singletons are ~5.58% of the ground truth. Correctly identifying singletons is critical since false merges score 0.0 for that entity.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Supervised Pairwise Classifier + Adaptive Entity Decision Engine  
**Core Innovation:** Multi-channel evidence tracking with rarest-token prioritization, channel-diversity-aware budgets, and address joint-key blocking — combined with stratified hard-negative mining across 4 collision types at a 1:6 ratio to produce a high-precision LightGBM-style ranking model with dynamic K ∈ [0, 8].

**High-Level Pipeline:**

```
Raw Test Data (1,732,544 S1 entities, 9,969,589 S2+S3 targets)
        │
        ▼
[V3 Multi-Channel Evidence Candidate Generator]
        │
        ▼
[32-Feature Extraction (features_v3.py)]
        │
        ▼
[HistGradientBoosting Ranking Classifier]
        │
        ▼
[Evidence-Driven Adaptive K Decision Engine]
        │
        ▼
[Output: matching_results.tsv + candidate_pairs.tsv]
```

---

## 3. Candidate Generation (Blocking)

The V3 candidate generator (`src/candidate_generator_v3.py`) uses **6 independent retrieval channels** with channel-aware budgets and evidence tracking per candidate:

| Channel | Key Construction | Posting Limit | Purpose |
|---|---|---|---|
| **Exact Normalized Name** | `normalize(name)` | 25 | Perfect / near-perfect name matches |
| **Sorted Token Slug** | `"".join(sorted(clean_tokens(name)))` | 25 | Word-order-invariant matching ("Hotel Royal" = "Royal Hotel") |
| **5-char Slug Prefix** | `slug[:5]` (slug len ≥ 5, posting ≤ 100) | 25 | Abbreviation and prefix matching |
| **Clean Name Tokens** | Individual rare tokens (corpus freq ≤ 3,000) | 25 | Rare discriminative name token matching |
| **Address Joint Key** | `house_num + first_street_word` | 25 | Near-unique physical address anchor |
| **Postal Code** | 5–6 character postal codes | 25 | Geographic clustering (US ZIP, India PIN) |

**Candidate pool cap:** Union of all channels, capped at **75 candidates per S1 entity**.

**Multi-Channel Support Filter:** Candidates supported by ≥ 2 independent channels receive a priority boost and are exempt from pool trimming. This ensures strong-evidence matches are never discarded.

**Recovery Layer:** If fewer than 5 candidates are found via primary channels, a broader trigram/prefix recovery pass activates to prevent true matches from being missed due to unusual naming.

**Results:**
- Total candidate pairs generated: **37,841,209**
- Average candidates per S1 entity: **21.8**
- Candidate recall on 10.32M catalog: **79.01%**
- Candidate containment (matches ⊆ candidates): **100%**

**How true matches were protected:** Rarest-token prioritization (tokens with corpus frequency < 3,000 are used first) prevents common tokens from flooding the posting lists. The sorted-token slug resolves word-order transpositions. Address joint keys provide near-unique physical anchors in large catalogs.

---

## 4. Matching Model

### Features Used (32 total — `src/features_v3.py`)

**Name features:**
- `name_ratio`: RapidFuzz token ratio similarity
- `name_token_sort_ratio`: Word-order-invariant token sort ratio
- `name_token_set_ratio`: Token set ratio (handles subset names)
- `name_suffix_stripped_sim`: Similarity after stripping legal suffixes (Ltd, Inc, Corp, Pvt, etc.)
- `name_char3gram_jaccard`: Character 3-gram Jaccard similarity
- `name_exact_match`: Binary exact normalized name flag
- `name_slug_exact_match`: Binary alphanumeric slug exact match flag
- `name_length_diff`: Normalized character length difference
- `name_idf_overlap`: Corpus-IDF-weighted token overlap (penalizes generic tokens)
- `name_idf_jaccard`: Corpus-IDF-weighted Jaccard similarity
- `name_generic_only_match`: Flag — true if the only matching tokens are all generic (high FP risk)

**Address features:**
- `addr_token_sort_ratio`: Address token sort ratio
- `addr_ratio`: Address string ratio
- `addr_house_num_agree`: House number agreement flag
- `addr_house_num_conflict`: House number conflict flag (strong FP signal)
- `addr_postal_agree`: Postal/ZIP code agreement flag
- `addr_postal_conflict`: Postal/ZIP code conflict flag
- `addr_state_conflict`: State/region conflict flag
- `addr_joint_match`: Address joint key (house_num + street_word) exact match flag
- `both_have_address`: Both records have non-empty address

**Cross-field / composite features:**
- `joint_name_addr`: Product of name ratio × address ratio
- `name_high_addr_conflict`: High name similarity but conflicting address (strong FP signal)
- `high_addr_diff_name`: High address similarity but diverging name (weak match signal)
- `s2_missing_addr`: Source 2 record has empty address (triggers elevated threshold)

**Evidence / retrieval features:**
- `retrieval_num_channels`: Number of independent retrieval channels that retrieved this candidate (higher = more reliable)
- `retrieval_best_rank`: Best rank achieved across all channels (lower = more relevant)
- `channel_diversity`: Entropy of retrieval channel contributions
- `transliteration_candidate`: Candidate is likely a transliteration (triggers adjusted threshold)

**Source indicators:**
- `is_source2`: Binary flag — candidate is from Source 2
- `is_source3`: Binary flag — candidate is from Source 3
- `has_postal_code`: Binary flag — candidate has a parseable postal code
- `country_match`: Binary flag — country fields agree

**Model type:** `HistGradientBoostingClassifier` (scikit-learn) — equivalent capability to LightGBM, fully CPU-optimized, no GPU required.

```
HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=50,
    max_depth=8,
    min_samples_leaf=25,
    l2_regularization=1.5,
    random_state=42,
)
```

**Training data:** Stratified hard-negative mining (`src/hard_negative_miner_v3.py`) at **1:6 negative-to-positive ratio** across 4 distinct collision types:
1. `same_name_diff_addr` — same business name, different physical location
2. `same_addr_diff_name` — same address (multi-tenant building), different business
3. `generic_collision` — matches only on generic tokens (restaurant, hotel, services, etc.)
4. `multi_channel_distractor` — retrieved by multiple channels but genuinely different entity

**Threshold selection:** F₀.₅-optimized on held-out validation split (entity-level, no leakage). Base threshold: **0.58** for standard pairs; **0.75** for S2 missing-address pairs; **0.38** for S3 transliteration candidates.

### Adaptive Entity Decision Engine (`src/ranking_model_v3.py`)

After pairwise scoring, an evidence-driven decision engine determines final matches:

1. **Rank** all candidates for an S1 entity by descending match probability P(match).
2. **Singleton gate**: If top candidate P < 0.50, predict empty (singleton).
3. **Dynamic K ∈ [0, 8]**: Start from top candidate; continue accepting matches while:
   - P(next) ≥ base threshold (0.58)
   - Score margin Δ ≤ 0.12 (tight margin for subsequent matches: 0.08)
   - No slope-drop termination (probability drop > 0.25 between adjacent candidates signals distractor boundary)
4. **Multi-channel bonus**: Candidates supported by ≥ 2 independent retrieval channels receive a +0.03 confidence boost before thresholding.
5. **Source-specific calibration**: S2 missing-address pairs require higher confidence (0.75); S3 transliteration candidates use a relaxed threshold (0.38).

---

## 5. Results & Error Analysis

### Official Leaderboard Result

| Metric | Value |
|---|---|
| **Official Leaderboard Score (F₀.₅)** | **0.786** |
| Submission | Laptop 2 V3 Production |
| Test S1 entities | 1,732,544 |
| Total predicted links | 4,993,255 |
| Average matches per S1 | 2.88 |
| Singletons predicted | 112,003 (6.46%) |
| Official validator | **PASS** |

> **Important Note:** The leaderboard score of **0.786** is the **official competition score** from the leaderboard portal. It is **distinct from** the offline validation score of **0.8091** (Entity Macro F₀.₅) computed on a held-out training split against the 10.32M full target catalog. The offline score is an upper-bound estimate on a validation subset; the official score is authoritative.

### Offline Validation (10.32M Target Catalog)

| Metric | V3 Production | Baseline (0.539 LB) |
|---|---|---|
| Candidate Recall | **79.01%** | 50.33% |
| Pair Precision | **0.9183** | 0.7159 |
| Pair Recall | **0.7103** | 0.4855 |
| Entity Macro F₀.₅ | **0.8091** | 0.5602 |
| Source 2 Macro F₀.₅ | **0.7912** | — |
| Source 3 Macro F₀.₅ | **0.7462** | — |

### Common False Positives (wrong merges)
- **70.8%** Partial name collisions with distant or conflicting address (e.g. "Raj Hotel Mumbai" matched with "Raj Hotel Delhi")
- **15.2%** Source 2 records with missing address — matched on name alone when a common name is shared
- **10.1%** Distinct businesses in shared commercial buildings (same address, similar generic name)
- **3.7%** Corporate branch/chain collisions (same brand, different state)

**Mitigation**: `name_high_addr_conflict` penalty feature; elevated S2 missing-address threshold (0.75); tight score margin (Δ ≤ 0.12).

### Common False Negatives (missed matches)
- **93.8%** Retrieval misses (posting list caps missed the true match entirely — candidate recall ceiling effect)
- **6.2%** Model decision misses (candidate was retrieved but scored below threshold — usually severe transliteration or acronym divergence)

**Mitigation**: Rarest-token prioritization resolves posting-list bottlenecks; address joint keys and sorted-slug slugs recover word-order and transliteration variants.

---

## 6. Conclusion

Team Arnav's V3 Business Entity Resolution pipeline achieves an official leaderboard score of **0.786** through three coordinated innovations: (1) a multi-channel evidence candidate generator that recovers 79% of true matches in a 9.97M-entity catalog using rarest-token prioritization and address joint keys; (2) a 32-feature HistGradientBoosting ranking classifier trained with stratified hard-negative mining at a 1:6 ratio across four realistic collision types; and (3) an evidence-driven adaptive decision engine with dynamic K ∈ [0, 8], source-specific calibration, and multi-channel confidence boosts. The key lesson learned is that candidate recall against the full-scale catalog is the dominant bottleneck — improving recall from 50% to 79% drove the largest performance gains, far exceeding the incremental improvements from model tuning alone.

---

## Appendix

### A. Code Artefacts

All source code ships under `code/business_entity_resolution/` in this submission zip.

**Entry points to reproduce output files:**

The V3 pipeline is executed via the main orchestration script. Before running, configure the dataset path in `src/config.py` (line 17: `DATASET_ROOT`).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train V3 model (on training data)
python src/main.py --mode train

# 3. Run full-test inference (generates output files)
python src/main.py --mode predict \
    --model backups/backup_20260927_074130_v3_0786_baseline/laptop2_v3_model.joblib \
    --output-dir output/

# 4. Validate output before submitting
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

**Source file summary:**

| File | Purpose |
|---|---|
| `src/candidate_generator_v3.py` | V3 multi-channel evidence candidate generator (primary blocking) |
| `src/features_v3.py` | 32-feature extraction for candidate pairs |
| `src/hard_negative_miner_v3.py` | Stratified hard-negative mining (4 collision types, 1:6 ratio) |
| `src/ranking_model_v3.py` | HistGradientBoosting classifier + adaptive entity decision engine |
| `src/data_loader.py` | TSV data loading and ground-truth parsing |
| `src/config.py` | Centralized path and hyperparameter configuration |
| `src/output.py` | Output file generation (matching_results.tsv, candidate_pairs.tsv) |
| `src/evaluate.py` | Offline F₀.₅ evaluation on held-out validation splits |
| `src/preprocessing.py` | Text normalization, legal suffix stripping, address expansion |

**Expected output:**
- `output/matching_results.tsv` — 1,732,545 lines (1 header + 1,732,544 data rows), ~86.8 MB
- `output/candidate_pairs.tsv` — 1,732,545 lines (1 header + 1,732,544 data rows), ~1.33 GB

**Inference runtime:** ~3.5 hours on CPU-only hardware (no GPU required), ~136 entities/second average throughput.

### B. Additional Results

**SHA256 Verification of Submitted Output Files:**

```
matching_results.tsv:  5288f38a13563cd9a4e67b74d5b75f34484691d362d9b140f65e291b0985f42e
candidate_pairs.tsv:   1ef68c822f5c997ee023c9b2fa4580f6e6eaaee8dc14fba0f0b4cc4cfe552895
```

**Official Validator Output:**
```
ML Challenge 2026 — submission validator
  test dir: C:\Arnav\dataset\student_resource\dataset\test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (112003 empty, 1620541 non-empty).
  candidate_pairs.tsv: 1732544 rows (1522 empty, 1731022 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
