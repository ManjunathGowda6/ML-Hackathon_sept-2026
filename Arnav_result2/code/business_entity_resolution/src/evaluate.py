"""
ML Challenge 2026 - Business Entity Resolution Pipeline
Evaluation & Validation Module

Computes exact macro-averaged F0.5 per Source 1 entity, including official singleton scoring:
- true == empty and pred == empty -> score 1.0 (correct singleton)
- true == empty and pred != empty -> score 0.0 (false merge)
- true != empty and pred == empty -> score 0.0 (missed match)
- true != empty and pred != empty -> F0.5 = (1.25 * P * R) / (0.25 * P + R)
"""

from typing import Dict, List, Set, Tuple
import numpy as np


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """
    Computes macro F0.5, Precision, and Recall across all Source 1 entities in ground truth.
    """
    scores = []
    precisions = []
    recalls = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if len(true_set) == 0:
            if len(pred_set) == 0:
                score = 1.0
                prec = 1.0
                rec = 1.0
            else:
                score = 0.0
                prec = 0.0
                rec = 1.0
        else:
            if len(pred_set) == 0:
                score = 0.0
                prec = 0.0
                rec = 0.0
            else:
                tp = len(true_set & pred_set)
                prec = tp / len(pred_set)
                rec = tp / len(true_set)
                denom = 0.25 * prec + rec
                score = (1.25 * prec * rec) / denom if denom > 0 else 0.0

        scores.append(score)
        precisions.append(prec)
        recalls.append(rec)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0

    return macro_f05, macro_prec, macro_rec


def run_validation(
    val_s1_records: List[Tuple[str, str, str, str]],
    blocking_index,
    matcher,
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float]:
    """
    Evaluate matcher on validation entities and report Macro F0.5.
    """
    print(f"\nEvaluating validation set ({len(val_s1_records):,} Source 1 entities)...", flush=True)
    from predict import predict_batch

    _, val_matches = predict_batch(val_s1_records, blocking_index, matcher)
    val_preds_set = {s1_id: set(m) for s1_id, m in val_matches.items()}

    # Filter ground truth to validation entities
    val_gt = {rec[0]: ground_truth.get(rec[0], set()) for rec in val_s1_records}

    f05, prec, rec = compute_macro_f05(val_gt, val_preds_set)

    print("\n" + "=" * 50, flush=True)
    print("Validation Results (Macro Averaged)", flush=True)
    print("=" * 50, flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"Macro F0.5: {f05:.4f} (Score: {f05 * 10:.2f} / 10)", flush=True)
    print("=" * 50 + "\n", flush=True)

    return f05, prec, rec
