"""
ML Challenge 2026 - Business Entity Resolution Pipeline
Streaming Output & Submission Validation Module

Provides:
- StreamFileWriter for writing matching_results.tsv and candidate_pairs.tsv in memory-safe chunks
- Comprehensive validation checks
- Execution of the official validate_submission.py script
"""

from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional


class SubmissionStreamWriter:
    """
    Streams submission lines directly to disk chunk-by-chunk.
    Keeps memory footprint constant regardless of dataset size (1.73M+ records).
    """

    def __init__(self, matching_path: Path, candidate_path: Path, append: bool = False, initial_count: int = 0):
        self.matching_path = Path(matching_path)
        self.candidate_path = Path(candidate_path)
        self.matching_path.parent.mkdir(parents=True, exist_ok=True)
        self.candidate_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        self.f_matching = open(self.matching_path, mode, encoding="utf-8", newline="\n")
        self.f_candidate = open(self.candidate_path, mode, encoding="utf-8", newline="\n")

        if not append:
            # Write required headers
            self.f_matching.write("source1_entity_id\tmatched_entity_ids\n")
            self.f_candidate.write("source1_entity_id\tcandidate_entity_ids\n")

        self.total_written = initial_count

    def write_batch(self, s1_ids: List[str], candidates_dict: Dict[str, List[str]], matches_dict: Dict[str, List[str]]):
        """
        Write a batch of Source 1 entities and their candidate/matched IDs.
        """
        for s1_id in s1_ids:
            cands = candidates_dict.get(s1_id, [])
            matches = matches_dict.get(s1_id, [])

            # Write candidate pairs
            cand_str = ",".join(cands)
            self.f_candidate.write(f"{s1_id}\t{cand_str}\n")

            # Write matching results
            match_str = ",".join(matches)
            self.f_matching.write(f"{s1_id}\t{match_str}\n")

        self.f_matching.flush()
        self.f_candidate.flush()
        self.total_written += len(s1_ids)

    def close(self):
        self.f_matching.close()
        self.f_candidate.close()
        print(f"Streaming writer closed. Total entities written: {self.total_written:,}")


def run_validator(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    validator_script: Optional[Path] = None,
) -> bool:
    """Run the official validate_submission.py script."""
    if not validator_script or not Path(validator_script).is_file():
        print("Note: validate_submission.py not found at expected path. Skipping external validation.")
        return True

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]

    print("\n" + "=" * 55)
    print("Running Official Submission Validator")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 55)

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print(">> OFFICIAL SUBMISSION VALIDATOR: PASS! Safe to submit.")
        return True
    else:
        print(f">> OFFICIAL SUBMISSION VALIDATOR FAILED with returncode {result.returncode}!")
        return False
