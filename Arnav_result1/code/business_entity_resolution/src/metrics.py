import numpy as np
from typing import Dict, Set, List

def compute_entity_macro_f05(ground_truth_dict: Dict[str, Set[str]],
                             predictions_dict: Dict[str, Set[str]]) -> dict:
    """
    Computes official entity-level Macro F0.5 across all S1 entities in ground_truth_dict.
    Singletons (0 true matches):
      - 0 predicted matches -> score = 1.0
      - >=1 predicted matches -> score = 0.0
    Non-singletons:
      - 0 predicted matches -> score = 0.0
      - Precision = TP / len(pred)
      - Recall = TP / len(gt)
      - F0.5 = 1.25 * P * R / (0.25 * P + R)
    """
    s1_scores = []
    s1_precisions = []
    s1_recalls = []
    
    total_tp = 0
    total_pred = 0
    total_gt = 0
    
    singletons_correct = 0
    singletons_total = 0
    
    pred_match_counts = []
    
    # Per source stats
    s2_tp, s2_pred, s2_gt = 0, 0, 0
    s3_tp, s3_pred, s3_gt = 0, 0, 0

    for s1_id, gt_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        pred_match_counts.append(len(pred_set))
        
        # Track GT by source
        for tid in gt_set:
            total_gt += 1
            if tid.startswith('S2-'): s2_gt += 1
            elif tid.startswith('S3-'): s3_gt += 1
            
        for tid in pred_set:
            total_pred += 1
            if tid.startswith('S2-'): s2_pred += 1
            elif tid.startswith('S3-'): s3_pred += 1
            
        tp_set = gt_set.intersection(pred_set)
        tp = len(tp_set)
        total_tp += tp
        for tid in tp_set:
            if tid.startswith('S2-'): s2_tp += 1
            elif tid.startswith('S3-'): s3_tp += 1
            
        if len(gt_set) == 0:
            singletons_total += 1
            if len(pred_set) == 0:
                s1_scores.append(1.0)
                s1_precisions.append(1.0)
                s1_recalls.append(1.0)
                singletons_correct += 1
            else:
                s1_scores.append(0.0)
                s1_precisions.append(0.0)
                s1_recalls.append(1.0)
        else:
            if len(pred_set) == 0:
                s1_scores.append(0.0)
                s1_precisions.append(0.0)
                s1_recalls.append(0.0)
            else:
                prec = tp / float(len(pred_set))
                rec = tp / float(len(gt_set))
                s1_precisions.append(prec)
                s1_recalls.append(rec)
                if tp == 0:
                    s1_scores.append(0.0)
                else:
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    s1_scores.append(f05)
                    
    macro_f05 = float(np.mean(s1_scores)) if s1_scores else 0.0
    macro_prec = float(np.mean(s1_precisions)) if s1_precisions else 0.0
    macro_rec = float(np.mean(s1_recalls)) if s1_recalls else 0.0
    
    pair_prec = total_tp / total_pred if total_pred > 0 else 0.0
    pair_rec = total_tp / total_gt if total_gt > 0 else 0.0
    pair_f05 = (1.25 * pair_prec * pair_rec) / (0.25 * pair_prec + pair_rec) if (0.25 * pair_prec + pair_rec) > 0 else 0.0
    
    pred_counts = np.array(pred_match_counts)
    
    return {
        'entity_macro_f05': macro_f05,
        'entity_macro_precision': macro_prec,
        'entity_macro_recall': macro_rec,
        'pair_precision': pair_prec,
        'pair_recall': pair_rec,
        'pair_f05': pair_f05,
        'total_tp': total_tp,
        'total_fp': total_pred - total_tp,
        'total_fn': total_gt - total_tp,
        'singletons_total': singletons_total,
        'singletons_correct': singletons_correct,
        'singleton_accuracy': (singletons_correct / singletons_total) if singletons_total > 0 else 1.0,
        'pred_matches_mean': float(np.mean(pred_counts)),
        'pred_matches_median': float(np.median(pred_counts)),
        'pred_matches_p90': float(np.percentile(pred_counts, 90)),
        'pred_matches_p95': float(np.percentile(pred_counts, 95)),
        'pred_matches_p99': float(np.percentile(pred_counts, 99)),
        'pred_matches_max': int(np.max(pred_counts)) if len(pred_counts) > 0 else 0,
        'source2_pair_precision': s2_tp / s2_pred if s2_pred > 0 else 0.0,
        'source2_pair_recall': s2_tp / s2_gt if s2_gt > 0 else 0.0,
        'source3_pair_precision': s3_tp / s3_pred if s3_pred > 0 else 0.0,
        'source3_pair_recall': s3_tp / s3_gt if s3_gt > 0 else 0.0,
    }
