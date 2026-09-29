"""
ML Challenge 2026 - Business Entity Resolution Pipeline
Configuration Module

Single source of truth for dataset paths, submission paths, and model settings.
"""

from pathlib import Path

# ==============================================================================
# 1. DATASET LOCATION CONFIGURATION
# ==============================================================================
# Only configure dataset root here. Do not hardcode anywhere else.
DATASET_ROOT = r"C:\Arnav\dataset"

# ==============================================================================
# 2. CANONICAL SUBMISSION & CANDIDATE PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_BACKUP_DIR = EXPERIMENTS_DIR / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical required submission path
SUBMISSION_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
EXPERIMENT_RESULTS_PATH = EXPERIMENTS_DIR / "results.csv"

# ==============================================================================
# 3. MODEL & MATCHING HYPERPARAMETERS
# ==============================================================================
MODEL_NAME = "dual_channel_exact_calibrated"
NAME_SIM_THRESHOLD = 0.84
ADDR_SIM_THRESHOLD = 0.74
JOINT_NAME_THRESHOLD = 0.55
JOINT_ADDR_THRESHOLD = 0.45
MAX_CANDIDATES_PER_S1 = 30
BATCH_CHUNK_SIZE = 50000
NEGATIVE_RATIO = 3
RANDOM_STATE = 42

# ==============================================================================
# 4. PATH RESOLUTION
# ==============================================================================
def resolve_dataset_paths(dataset_root_str: str):
    """Resolve paths for train and test TSV files."""
    root = Path(dataset_root_str)
    candidates = [
        (root / "train", root / "test"),
        (root / "student_resource" / "dataset" / "train", root / "student_resource" / "dataset" / "test"),
        (root / "dataset" / "train", root / "dataset" / "test"),
    ]

    resolved_train = None
    resolved_test = None

    for train_cand, test_cand in candidates:
        if train_cand.is_dir() and test_cand.is_dir():
            resolved_train = train_cand
            resolved_test = test_cand
            break

    if resolved_train is None or resolved_test is None:
        resolved_train = root / "train"
        resolved_test = root / "test"

    paths = {
        "train_dir": resolved_train,
        "test_dir": resolved_test,
        "train_source1": resolved_train / "train_source1.tsv",
        "train_source2": resolved_train / "train_source2.tsv",
        "train_source3": resolved_train / "train_source3.tsv",
        "train_ground_truth": resolved_train / "train_ground_truth.tsv",
        "test_source1": resolved_test / "test_source1.tsv",
        "test_source2": resolved_test / "test_source2.tsv",
        "test_source3": resolved_test / "test_source3.tsv",
    }
    return paths

DATASET_PATHS = resolve_dataset_paths(DATASET_ROOT)

def find_validator_script():
    possible_paths = [
        PROJECT_ROOT / "utils" / "validate_submission.py",
        Path(DATASET_ROOT) / "student_resource" / "utils" / "validate_submission.py",
        Path(DATASET_ROOT).parent / "student_resource" / "utils" / "validate_submission.py",
    ]
    for p in possible_paths:
        if p.is_file():
            return p
    return None

VALIDATOR_SCRIPT_PATH = find_validator_script()
