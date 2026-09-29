# Business Entity Resolution

## Laptop 1 — Best Offline Validation Result

### Overview
This package contains the source code, offline evaluation benchmarks, and artifact structure for the **Laptop 1 V5-Anchored (V2 Model)** Business Entity Resolution pipeline.

### Model Architecture
- **Retrieval Engine**: Dual Multi-Channel Inverted Indices (Primary multi-channel retriever + Selective secondary recall recovery engine).
- **Representation**: Multi-View Business Representation (Original, compact, domain-stem, legal-suffix-stripped, character n-grams, sorted-token, postal/address normalizations).
- **Pair Scorer**: 71-feature LightGBM Multi-View Pair Matcher.
- **Decision Layer**: Precision-Calibrated Gate with adaptive margins, severe conflict vetoes (country, postal, house number), commercial building collision protection, and strict $K \le 8$ candidate capping.

---

### Validation Results (Offline Benchmark)

#### Primary Benchmark (Validation A — 3,000 S1 Entities / 509,250 Target Universe)
- **Entity Macro F0.5**: **0.8791**
- **Entity Precision**: **0.9134**
- **Entity Recall**: **0.8537**
- **Pair Precision**: **0.9188**
- **Pair Recall**: **0.8441**
- **Candidate Recall**: **94.33%**

#### Independent Anti-Overfitting Validation (Dual-Slice Benchmark)
- **Slice A (Queries 1–1,500) Macro F0.5**: **0.8641**
- **Slice B (Queries 1,501–3,000) Macro F0.5**: **0.8626**

#### Measured Improvement
- **+0.0422 Macro F0.5** versus the reproduced V5 baseline of `0.8369`
- **+0.0602 Macro F0.5** versus the earlier `0.8189` reference baseline

---

### CRITICAL LIMITATION & CLARIFICATION
> **IMPORTANT:**
> - The **0.8791** score is strictly an **OFFLINE VALIDATION / EXPERIMENTAL RESULT** evaluated on held-out reference splits.
> - It was **NOT** confirmed by a competition leaderboard submission.
> - **DO NOT describe or cite 0.8791 as the leaderboard score.**
> - The known proven competition leaderboard result is separate: **Laptop 2 V3 leaderboard score = 0.786**.

---

### Execution & Production Status Documentation
1. **Full-Test Fresh V2 Inference**: Full-test fresh inference across all 1,732,544 test S1 entities was not completed to the end.
2. **Production SHA Audit**: The previously generated Laptop 1 full-test production files (`matching_results.tsv` SHA256: `908821E3...`) shared earlier outputs and were therefore not accepted as proof of 0.8791 leaderboard performance.
3. **Fresh V2 Checkpoint**: A later fresh V2 full-test run (`run_fresh_v2_full_test.py`) was launched and safely checkpointed after **9,000 / 1,732,544 S1 entities** (`PRAGMA integrity_check: ok` in `fresh_v2_pipeline.db`) due to the hard time limit.
4. **Integrity & Claims**: No unsupported claims are made. The files and metrics presented here reflect strictly reproducible offline validation benchmarks.

---

### Source Code Structure (`src/`)
- `multiview.py`: Multi-view business record representations and text normalizers.
- `retrieval.py`: Primary fast multi-channel inverted index retrieval engine.
- `recall_recovery.py`: Selective secondary recall recovery retriever and candidate merger.
- `features.py`: 71-dimensional multi-view pair feature vector extraction.
- `graph_ranker.py`: Entity-level decision policies and candidate rankers.
- `metrics.py`: Official Entity Macro F0.5, Pair Precision/Recall, and candidate metrics.
- `benchmark_two_slices.py`: Independent dual-slice anti-overfitting evaluation harness.
- `evaluate_v5_anchored.py`: Forensic evaluation harness across all ablation modes.
- `deep_forensic_error_mining.py`: Categorization of remaining false negatives and false positives.
- `run_fresh_v2_full_test.py`: Production inference pipeline with resumable SQLite checkpointing.
- `gpu_cuda_benchmark.py`: CUDA tensor execution benchmark for GTX 1650.
