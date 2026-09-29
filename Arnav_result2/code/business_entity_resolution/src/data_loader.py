"""
ML Challenge 2026 - Business Entity Resolution Pipeline
Optimized Data Loader Module

Loads all TSV datasets using `pd.read_csv(path, sep="\\t")` as required by the hackathon.
Provides high-speed streaming iterators and target loaders operating at >150k rec/s.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple
import pandas as pd


def load_tsv(file_path: Path, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load a TSV dataset using pandas pd.read_csv(path, sep='\\t')."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Required TSV file not found: {file_path}")

    if sample_n is not None:
        df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False, nrows=sample_n)
    else:
        df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)

    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def load_target_tuples(tsv_path: Path, sample_n: Optional[int] = None) -> List[Tuple[str, str, str, str]]:
    """
    High-speed parser loading target records into compact tuples:
    (entity_id, name_lower, address_lower, country_lower)
    """
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if sample_n is not None and i >= sample_n:
                break
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            name = parts[1].lower() if len(parts) > 1 else ""
            addr = parts[2].lower() if len(parts) > 2 else ""
            country = parts[3].lower() if len(parts) > 3 else ""
            records.append((eid, name, addr, country))
    return records


def stream_s1_batches(
    tsv_path: Path,
    batch_size: int = 50000,
    max_rows: Optional[int] = None,
    skip_rows: int = 0,
) -> Generator[List[Tuple[str, str, str, str]], None, None]:
    """
    Streaming generator yielding batches of Source 1 records as:
    (s1_id, name_lower, address_lower, country_lower)
    Supports skip_rows for resuming interrupted inference runs.
    """
    current_batch = []
    total_yielded = 0
    total_skipped = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            if total_skipped < skip_rows:
                total_skipped += 1
                continue
            if max_rows is not None and total_yielded >= max_rows:
                break
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            name = parts[1].lower() if len(parts) > 1 else ""
            addr = parts[2].lower() if len(parts) > 2 else ""
            country = parts[3].lower() if len(parts) > 3 else ""

            current_batch.append((eid, name, addr, country))
            total_yielded += 1

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

    if current_batch:
        yield current_batch


def load_ground_truth(file_path: Path, filter_s1_ids: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Load ground truth mappings {s1_id: set(matched_ids)}."""
    ground_truth = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if filter_s1_ids is not None and s1_id not in filter_s1_ids:
                continue
            matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
            ground_truth[s1_id] = matches
    return ground_truth
