# Business Entity Resolution Pipeline — ML Challenge 2026

A modular, robust, reproducible, and high-performance Machine Learning solution for the **ML Challenge 2026: Business Entity Resolution Hackathon**.

---

## 1. Problem Overview
In commercial entity databases, business records originate from disparate, noisy sources without shared primary keys.
- **Source 1 (`S1-`)**: Reference master business dataset.
- **Source 2 (`S2-`) & Source 3 (`S3-`)**: Potentially matching business records containing noisy names, partial addresses, landmark references, abbreviations, and regional variations.
- **Goal**: Identify which records in Source 2 and Source 3 represent the same real-world business entity as each Source 1 record.
- **Singletons**: A Source 1 business entity may match **zero, one, or multiple** records from Source 2 / Source 3. Correctly identifying singletons (predicting empty matches) receives full evaluation credit (1.0).
- **Target Metric**: Macro-averaged **$F_{0.5}$ score** across all Source 1 entities, which penalizes false positive merges twice as heavily as false negative omissions.

---

## 2. Project Architecture

```
business_entity_resolution/
│
├── src/
│   ├── config.py           # Unified settings, dataset root, hyperparameters
│   ├── data_loader.py      # TSV loader (pd.read_csv(sep='\t')) & ground truth parsing
│   ├── preprocessing.py    # Reusable text normalization, legal suffixes, address expansion
│   ├── blocking.py         # Multi-rule inverted index candidate generation (Rules A-D)
│   ├── features.py         # 13 interpretable string, token, exact, & char n-gram features
│   ├── train.py            # Pair construction (NEGATIVE_RATIO:1) & classifier training
│   ├── evaluate.py         # Leakage-free entity split, macro F0.5, threshold optimization
│   ├── predict.py          # Probability scoring, thresholding & singleton handling
│   ├── output.py           # Generation of matching_results.tsv & candidate_pairs.tsv
│   └── main.py             # 8-stage pipeline orchestrator
│
├── output/
│   ├── matching_results.tsv  # Final matches (scored on leaderboard)
│   └── candidate_pairs.tsv   # Candidate pairs fed to ML model
│
├── models/
│   ├── er_model.joblib       # Trained ML classifier artifact
│   └── best_threshold.txt    # Optimized F0.5 decision threshold
│
├── utils/
│   └── validate_submission.py# Official hackathon submission validator
│
├── requirements.txt        # Pinned dependencies
├── Documentation_template.md # Methodology write-up for submission zip
└── README.md               # Complete end-to-end documentation
```

---

## 3. Dataset Configuration

All dataset paths are centralized in **one single file**: [`src/config.py`](file:///c:/Arnav/business_entity_resolution/src/config.py).

### Expected Dataset Folder Structure:
```
DATASET_ROOT/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Setting Dataset Path:
Open [`src/config.py`](file:///c:/Arnav/business_entity_resolution/src/config.py) and configure your path at line 17:
```python
DATASET_ROOT = r"C:\Arnav\dataset"
```
> **Smart Path Resolution**: The pipeline automatically checks both `DATASET_ROOT/train` and `DATASET_ROOT/student_resource/dataset/train` seamlessly, meaning you never have to adjust paths across modules.

---

## 4. Pipeline Walkthrough

### Stage 1: Data Loading & Handling (`src/data_loader.py`)
- All files are loaded via `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
- Handles empty strings, missing values, Unicode characters, and singletons.
- Ground truth mappings `{s1_id: set(matched_ids)}` are parsed with streaming support.

### Stage 2: Text Normalization (`src/preprocessing.py`)
- **Unicode normalization**: Replaces accented glyphs (NFKD).
- **Symbol normalization**: `&` $\to$ `and`, `@` $\to$ `at`, `+` $\to$ `plus`.
- **Legal corporate suffix cleaning**: Normalizes or removes suffixes (`pvt ltd`, `private limited`, `llc`, `inc`, `corp`, `gmbh`, `sarl`, `sa`, `llp`, `pc`, etc.) when distinguishing name content remains:
  - `"ABC Pvt. Ltd."` $\to$ `"abc"`
  - `"ABC Private Limited"` $\to$ `"abc"`
- **Address abbreviation expansion**: `rd` $\to$ `road`, `st` $\to$ `street`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `apt` $\to$ `unit`, `opp` $\to$ `opposite`, `nr` $\to$ `near`.
- **Field extraction**: Extracts postal PIN/ZIP codes (5 or 6 digits) and building/house numbers.
- **Country normalization**: Supports arbitrary country labels (US, India, France, etc.). Original columns are preserved untouched, adding `name_norm`, `address_norm`, `country_norm`, `postal_code`, and `house_number`.

### Stage 3: Candidate Generation & Blocking (`src/blocking.py`)
Comparing all pairs directly would require $1.7\text{M} \times 10\text{M} = 17\text{ trillion}$ comparisons. The blocking engine builds an inverted index over Source 2 and Source 3 in $O(1)$ token lookup time:
- **Rule A (Same country + name tokens)**: Matches on significant business name tokens.
- **Rule B (Same country + address tokens)**: Matches on distinctive address tokens.
- **Rule C (Strong name match)**: Matches exact names or distinctive first name tokens regardless of country or missing address.
- **Rule D (Shared address tokens)**: Matches identical postal/PIN codes or house numbers.
- **Union & Ranking**: Combines candidate matches across all rules, ranks them by multi-rule trigger count, deduplicates pairs, and retains top `MAX_CANDIDATES_PER_S1 = 30` candidates per entity.

### Stage 4: Feature Engineering (`src/features.py`)
For each candidate pair, 13 tabular features are extracted:
1. `name_similarity`: Levenshtein ratio via `rapidfuzz` (with pure-Python fallback).
2. `name_token_jaccard`: Word token intersection over union.
3. `name_length_difference`: Relative normalized length difference.
4. `address_similarity`: Fuzzy Levenshtein ratio on normalized addresses.
5. `address_token_jaccard`: Word token Jaccard similarity on addresses.
6. `address_length_difference`: Relative normalized length difference on addresses.
7. `country_same`: Indicator (1.0 if identical non-empty, 0.0 otherwise).
8. `name_exact_match`: Exact match indicator.
9. `address_exact_match`: Exact address match indicator.
10. `same_postal_code`: Match indicator on extracted postal/PIN code.
11. `same_house_number`: Match indicator on extracted house/building number.
12. `name_char_cosine`: Character 3-gram cosine similarity on business names.
13. `address_char_cosine`: Character 3-gram cosine similarity on addresses.

### Stage 5: Labeled Pair Construction & Model Training (`src/train.py`)
- **Positives**: Ground-truth links.
- **Negatives**: Hard negatives sampled from candidate pairs generated by the blocking stage (not random noise).
- **Ratio**: Configurable `NEGATIVE_RATIO = 3` (3 negatives per positive).
- **Classifier**: Baseline `LogisticRegression(class_weight='balanced')`, with optional support for `RandomForestClassifier` and `HistGradientBoostingClassifier`.

### Stage 6: Validation & Threshold Search (`src/evaluate.py`)
- **No Data Leakage**: Source 1 entities are split at the **entity level** (80% train, 20% validation) so no Source 1 entity appears in both training and evaluation.
- **Macro $F_{0.5}$ Evaluation**:
  $$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{i \in S_1} \frac{1.25 \times P_i \times R_i}{0.25 \times P_i + R_i}$$
  - Correct singletons (empty predicted, empty ground truth) score **1.0**.
  - False positive merges on singletons score **0.0**.
- **Threshold Grid Search**: Evaluates candidate decision thresholds $[0.50, 0.55, 0.60, \dots, 0.95]$ and selects the threshold maximizing Macro $F_{0.5}$.

### Stage 7: Test Inference & Singleton Handling (`src/predict.py`)
- Computes match probabilities for all test candidate pairs.
- Retains matches where $P(\text{match}) \ge \text{MATCH\_THRESHOLD}$.
- If no candidate passes the threshold, the match list remains empty (`[]`), guaranteeing correct singleton predictions.

### Stage 8: Output TSV & Official Validation (`src/output.py`)
- Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Formatted as UTF-8, tab-separated, comma-delimited ID lists without quotes.
- Guarantees **every single test Source 1 entity** appears exactly once.
- Validates that all final matches are a strict subset of candidate pairs.
- Automatically executes the official `utils/validate_submission.py` script.

---

## 5. Installation & Requirements

### Environment Requirements
- Python 3.8+ (tested on Python 3.10)
- Operating System: Windows, Linux, or macOS

### Install Dependencies
```bash
cd business_entity_resolution
pip install -r requirements.txt
```

---

## 6. How to Run

### Run Full Pipeline
To execute the complete end-to-end pipeline:
```bash
python src/main.py
```

### Run on a Sample for Rapid Testing
To test the pipeline on a subset of entities (e.g. 500 or 5,000 records):
```bash
python src/main.py --sample 500
```

### Select Alternative ML Classifiers
```bash
python src/main.py --model random_forest
# or
python src/main.py --model hist_gradient_boosting
```

### Manually Run the Official Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../dataset/student_resource/dataset/test
```

---

## 7. Submission Verification Checklist

- [x] Configurable `DATASET_ROOT` in [`src/config.py`](file:///c:/Arnav/business_entity_resolution/src/config.py)
- [x] All TSVs read with `pd.read_csv(sep="\t")`
- [x] No Source 1 $\to$ Source 1 matches
- [x] Every test Source 1 entity appears exactly once
- [x] Singletons correctly represented as empty strings
- [x] Final matches are a strict subset of candidate pairs
- [x] No external APIs, external databases, or geocoders used
- [x] Generalizes to arbitrary countries (US, India, France, etc.)
- [x] Official `validate_submission.py` passes with **PASS** status
