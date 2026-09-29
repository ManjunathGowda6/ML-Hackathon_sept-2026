import numpy as np
from typing import List, Dict, Tuple

class EntityMatchDecider:
    def __init__(self,
                 min_top_score: float = 0.36,
                 multi_match_threshold: float = 0.42,
                 max_gap_from_top: float = 0.14,
                 s2_penalty: float = 0.02):
        self.min_top_score = min_top_score
        self.multi_match_threshold = multi_match_threshold
        self.max_gap_from_top = max_gap_from_top
        self.s2_penalty = s2_penalty

    def decide_matches_for_entity(self, candidate_records: List[dict]) -> List[str]:
        """
        Takes a list of candidate dictionaries for a single S1 entity, each containing:
        - target_id
        - raw_score (from model)
        - is_source_2
        - postal_conflict
        - country_conflict
        - house_num_conflict
        - name_exact_match
        - channel_hit_count
        
        Returns a list of accepted target_ids (empty list for singleton).
        """
        if not candidate_records:
            return []
            
        # Adjust raw score with calibrated source / conflict penalties
        scored_candidates = []
        for c in candidate_records:
            adj_score = c['raw_score']
            
            # Source 2 is noisier, apply small calibration penalty
            if c.get('is_source_2', 0) > 0.5:
                adj_score -= self.s2_penalty
                
            # Severe conflict penalties
            if c.get('country_conflict', 0) > 0.5:
                adj_score -= 0.40  # Different country is almost certainly a false positive
                
            if c.get('postal_conflict', 0) > 0.5 and c.get('name_exact_match', 0) < 0.5:
                adj_score -= 0.25  # Different postal code without exact name match
                
            if c.get('house_num_conflict', 0) > 0.5 and c.get('name_exact_match', 0) < 0.5:
                adj_score -= 0.15  # Different house number
                
            scored_candidates.append({
                'target_id': c['target_id'],
                'adj_score': adj_score,
                'raw_score': c['raw_score'],
                'meta': c
            })
            
        # Sort candidates by adjusted score descending
        scored_candidates.sort(key=lambda x: x['adj_score'], reverse=True)
        
        top = scored_candidates[0]
        if top['adj_score'] < self.min_top_score:
            # Evidence insufficient -> declare singleton (0 matches)
            return []
            
        accepted_ids = [top['target_id']]
        top_score = top['adj_score']
        
        # Check subsequent candidates for legitimate multi-matches
        for cand in scored_candidates[1:]:
            c_score = cand['adj_score']
            gap = top_score - c_score
            
            # Conditions for accepting an additional match:
            # 1. Candidate must exceed multi_match_threshold
            # 2. Score gap to top candidate must be within max_gap_from_top
            # 3. No conflicting country or postal code
            meta = cand['meta']
            if (c_score >= self.multi_match_threshold and
                gap <= self.max_gap_from_top and
                meta.get('country_conflict', 0) < 0.5 and
                meta.get('postal_conflict', 0) < 0.5):
                accepted_ids.append(cand['target_id'])
            else:
                # Once scores drop significantly, stop scanning
                if gap > self.max_gap_from_top + 0.10:
                    break
                    
        return accepted_ids
