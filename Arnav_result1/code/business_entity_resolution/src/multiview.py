import re
import unicodedata
from rapidfuzz import fuzz
try:
    import anyascii
except ImportError:
    anyascii = None

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(?:pvt|private|ltd|limited|inc|incorporated|corp|corporation|llc|llp|co|company|'
    r'gmbh|ag|sa|sas|sarl|eurl|sci|ste|snc|bv|nv|plc|lp|holdings|group|enterprises|enterprise)\b',
    re.IGNORECASE
)

ADDRESS_ABBREV_MAP = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bfl\b': 'floor',
    r'\bflr\b': 'floor',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bno\b': 'number',
    r'\bnear\b': 'near',
    r'\bopp\b': 'opposite',
    r'\bmkt\b': 'market',
    r'\bindl\b': 'industrial',
    r'\bind\b': 'industrial',
    r'\bsec\b': 'sector',
    r'\bdist\b': 'district',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bpobox\b': 'pobox',
    r'\bp\.?o\.?\s*box\b': 'pobox',
    r'\bhwy\b': 'highway',
    r'\bdept\b': 'department',
    r'\bintl\b': 'international',
    r'\btech\b': 'technology',
    r'\bmfg\b': 'manufacturing',
}

PIN_US = re.compile(r'\b\d{5}(?:-\d{4})?\b')
PIN_IN = re.compile(r'\b\d{6}\b')
PIN_FR = re.compile(r'\b\d{5}\b')
HOUSE_NUM_RE = re.compile(r'\b(?:no\.?|plot|flat|shop|door|suite|ste|#)?\s*([0-9]+[a-z]?)\b', re.IGNORECASE)
DOMAIN_RE = re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9\-]+)(?:\.(?:com|in|org|net|fr|co|info|biz|io|gov|edu))\b', re.IGNORECASE)

GENERIC_ADDR_TOKENS = {
    'street', 'road', 'avenue', 'lane', 'drive', 'court', 'floor', 'suite', 'apartment',
    'number', 'near', 'opposite', 'market', 'industrial', 'sector', 'district', 'place',
    'square', 'pobox', 'house', 'plot', 'flat', 'shop', 'door', 'building', 'bldg',
    'north', 'south', 'east', 'west', 'road', 'cross', 'main', 'first', 'second',
    'india', 'united', 'states', 'france', 'california', 'texas', 'florida', 'york',
    'maharashtra', 'delhi', 'tamil', 'nadu', 'karnataka', 'telangana', 'gujarat'
}

def strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def extract_domain_stem(s: str) -> str:
    if not s:
        return ""
    m = DOMAIN_RE.search(s)
    if m:
        return m.group(1).lower().replace('-', '')
    return ""

def transliterate_text(s: str) -> str:
    if not s:
        return ""
    if anyascii is not None and not s.isascii():
        return anyascii.anyascii(s)
    return s

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = strip_accents(s).lower()
    dom_stem = extract_domain_stem(s)
    if dom_stem:
        s = s + " " + dom_stem
    s = re.sub(r'&', ' and ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compact_text(s: str) -> str:
    if not s:
        return ""
    s_clean = strip_accents(s).lower()
    # Check domain stem directly
    dom_stem = extract_domain_stem(s_clean)
    if dom_stem and len(dom_stem) >= 3:
        return dom_stem
    # Remove domain suffixes if present
    s_clean = re.sub(r'(?:\.(?:com|in|org|net|fr|co|info|biz|io|gov|edu))\b', '', s_clean)
    return re.sub(r'[^a-z0-9]', '', s_clean)

def extract_legal_suffix(s_norm: str):
    matches = LEGAL_SUFFIXES_REGEX.findall(s_norm)
    return set(matches) if matches else set()

def strip_legal_suffixes(s_norm: str) -> str:
    stripped = LEGAL_SUFFIXES_REGEX.sub(' ', s_norm)
    return re.sub(r'\s+', ' ', stripped).strip()

def sorted_tokens(s_norm: str) -> str:
    toks = [t for t in s_norm.split() if len(t) > 0]
    toks.sort()
    return " ".join(toks)

def normalize_address(s: str) -> str:
    if not s:
        return ""
    s_norm = normalize_text(s)
    for pattern, repl in ADDRESS_ABBREV_MAP.items():
        s_norm = re.sub(pattern, repl, s_norm)
    s_norm = re.sub(r'\s+', ' ', s_norm).strip()
    return s_norm

def extract_postal_code(s_raw: str, country: str = ""):
    if not s_raw:
        return ""
    country = str(country).upper()
    if country == "INDIA" or country == "IN":
        m = PIN_IN.findall(s_raw)
        if m: return m[-1]
    elif country == "FRANCE" or country == "FR":
        m = PIN_FR.findall(s_raw)
        if m: return m[-1]
    else:
        m = PIN_US.findall(s_raw)
        if m: return m[-1][:5]
        m2 = PIN_IN.findall(s_raw)
        if m2: return m2[-1]
    return ""

def extract_house_number(s_norm: str):
    if not s_norm:
        return ""
    m = HOUSE_NUM_RE.search(s_norm)
    if m:
        return m.group(1)
    return ""

class MultiViewRecord:
    __slots__ = (
        'entity_id', 'country', 'raw_name', 'norm_name', 'compact_name',
        'legal_stripped_name', 'compact_stripped_name', 'legal_suffixes',
        'tokens_name', 'sorted_name', 'domain_stem', 'char_3grams_name',
        'translit_name', 'compact_translit_name', 'tokens_translit_name',
        'raw_address', 'norm_address', 'compact_address', 'tokens_address',
        'rare_tokens_address', 'sorted_address', 'postal_code', 'house_number',
        'translit_address', 'tokens_translit_address'
    )
    
    def __init__(self, entity_id: str, business_name: str, business_address: str, country: str):
        self.entity_id = entity_id
        self.country = str(country).strip().upper() if country else ""
        self.raw_name = str(business_name) if business_name and str(business_name) != 'nan' else ""
        self.raw_address = str(business_address) if business_address and str(business_address) != 'nan' else ""
        
        # Domain stem extraction
        self.domain_stem = extract_domain_stem(self.raw_name)

        # Transliteration views (Indic/Non-ASCII)
        if not self.raw_name.isascii():
            raw_tname = transliterate_text(self.raw_name)
            self.translit_name = normalize_text(raw_tname)
            self.compact_translit_name = compact_text(self.translit_name)
            self.tokens_translit_name = [t for t in self.translit_name.split() if len(t) > 0]
        else:
            self.translit_name = normalize_text(self.raw_name)
            self.compact_translit_name = compact_text(self.translit_name)
            self.tokens_translit_name = [t for t in self.translit_name.split() if len(t) > 0]
        
        # Name views
        self.norm_name = normalize_text(self.raw_name)
        if len(self.norm_name) < 2 and len(self.translit_name) >= 2:
            self.norm_name = self.translit_name
            
        self.compact_name = compact_text(self.norm_name)
        if len(self.compact_name) < 2 and len(self.compact_translit_name) >= 2:
            self.compact_name = self.compact_translit_name

        self.legal_stripped_name = strip_legal_suffixes(self.norm_name)
        self.compact_stripped_name = compact_text(self.legal_stripped_name)
        self.legal_suffixes = extract_legal_suffix(self.norm_name)
        self.tokens_name = [t for t in self.norm_name.split() if len(t) > 0]
        if self.domain_stem and self.domain_stem not in self.tokens_name:
            self.tokens_name.append(self.domain_stem)
        self.sorted_name = sorted_tokens(self.norm_name)

        # Character 3-grams
        cname = self.compact_stripped_name or self.compact_name or self.compact_translit_name
        if len(cname) >= 3:
            self.char_3grams_name = {cname[i:i+3] for i in range(len(cname)-2)}
        else:
            self.char_3grams_name = set()

        # Address Transliteration
        if not self.raw_address.isascii():
            raw_taddr = transliterate_text(self.raw_address)
            self.translit_address = normalize_address(raw_taddr)
            self.tokens_translit_address = [t for t in self.translit_address.split() if len(t) > 0]
        else:
            self.translit_address = normalize_address(self.raw_address)
            self.tokens_translit_address = [t for t in self.translit_address.split() if len(t) > 0]

        # Address views
        self.norm_address = normalize_address(self.raw_address)
        if len(self.norm_address) < 3 and len(self.translit_address) >= 3:
            self.norm_address = self.translit_address

        self.compact_address = compact_text(self.norm_address)
        self.tokens_address = [t for t in self.norm_address.split() if len(t) > 0]
        self.rare_tokens_address = [t for t in self.tokens_address if len(t) >= 4 and t not in GENERIC_ADDR_TOKENS and not t.isdigit()]
        self.sorted_address = sorted_tokens(self.norm_address)
        self.postal_code = extract_postal_code(self.raw_address, self.country)
        self.house_number = extract_house_number(self.norm_address)

def compute_multiview_pair_features(r1: MultiViewRecord, r2: MultiViewRecord, idf_dict=None, cand_evidence=None):
    feats = {}
    
    # 1. Exact match signals
    feats['name_exact_match'] = 1.0 if r1.norm_name and r1.norm_name == r2.norm_name else 0.0
    feats['name_compact_exact'] = 1.0 if r1.compact_name and (
        r1.compact_name == r2.compact_name or 
        r1.compact_stripped_name == r2.compact_stripped_name or
        (r1.compact_stripped_name and r1.compact_stripped_name == r2.compact_name) or
        (r2.compact_stripped_name and r2.compact_stripped_name == r1.compact_name)
    ) else 0.0
    feats['name_stripped_exact'] = 1.0 if r1.legal_stripped_name and r1.legal_stripped_name == r2.legal_stripped_name else 0.0
    feats['name_sorted_exact'] = 1.0 if r1.sorted_name and r1.sorted_name == r2.sorted_name else 0.0
    
    # 2. String similarities
    feats['name_ratio'] = fuzz.ratio(r1.norm_name, r2.norm_name) / 100.0
    feats['name_partial_ratio'] = fuzz.partial_ratio(r1.norm_name, r2.norm_name) / 100.0
    feats['name_token_sort_ratio'] = fuzz.token_sort_ratio(r1.norm_name, r2.norm_name) / 100.0
    feats['name_token_set_ratio'] = fuzz.token_set_ratio(r1.norm_name, r2.norm_name) / 100.0
    feats['name_stripped_ratio'] = fuzz.ratio(r1.legal_stripped_name, r2.legal_stripped_name) / 100.0 if (r1.legal_stripped_name and r2.legal_stripped_name) else 0.0
    
    # Transliterated name similarities
    feats['translit_name_ratio'] = fuzz.ratio(r1.translit_name, r2.translit_name) / 100.0
    feats['translit_token_sort_ratio'] = fuzz.token_sort_ratio(r1.translit_name, r2.translit_name) / 100.0

    # 3. Token Jaccard & Containment
    set1, set2 = set(r1.tokens_name), set(r2.tokens_name)
    inter = set1.intersection(set2)
    union = set1.union(set2)
    feats['name_token_jaccard'] = len(inter) / len(union) if union else 0.0
    feats['name_token_containment'] = len(inter) / min(len(set1), len(set2)) if set1 and set2 else 0.0

    # Transliterated Token Jaccard
    tset1, tset2 = set(r1.tokens_translit_name), set(r2.tokens_translit_name)
    tinter = tset1.intersection(tset2)
    tunion = tset1.union(tset2)
    feats['translit_name_jaccard'] = len(tinter) / len(tunion) if tunion else 0.0

    # 4. Domain & Character Signatures
    stem1, stem2 = r1.domain_stem, r2.domain_stem
    if stem1 or stem2:
        c1 = r1.compact_stripped_name or r1.compact_name
        c2 = r2.compact_stripped_name or r2.compact_name
        feats['domain_stem_match'] = 1.0 if (
            (stem1 and stem1 == stem2) or
            (stem1 and stem1 == c2) or
            (stem2 and stem2 == c1)
        ) else 0.0
    else:
        feats['domain_stem_match'] = 0.0

    # Character 3-gram Jaccard
    ng1, ng2 = r1.char_3grams_name, r2.char_3grams_name
    ng_inter = ng1.intersection(ng2)
    ng_union = ng1.union(ng2)
    feats['char_3gram_jaccard'] = len(ng_inter) / len(ng_union) if ng_union else 0.0

    # 5. IDF / Rarity weighted token overlap
    if idf_dict:
        overlap_idf = sum(idf_dict.get(t, 1.0) for t in inter)
        total_idf1 = sum(idf_dict.get(t, 1.0) for t in set1)
        total_idf2 = sum(idf_dict.get(t, 1.0) for t in set2)
        feats['name_weighted_overlap'] = (2.0 * overlap_idf) / (total_idf1 + total_idf2) if (total_idf1 + total_idf2) > 0 else 0.0
        feats['name_max_overlap_idf'] = max([idf_dict.get(t, 1.0) for t in inter], default=0.0)
    else:
        feats['name_weighted_overlap'] = feats['name_token_jaccard']
        feats['name_max_overlap_idf'] = 0.0
        
    len1, len2 = len(r1.norm_name), len(r2.norm_name)
    feats['name_len_diff'] = abs(len1 - len2)
    feats['name_len_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # 6. Legal suffix compatibility
    suf1, suf2 = r1.legal_suffixes, r2.legal_suffixes
    if suf1 and suf2:
        feats['legal_suffix_compatible'] = 1.0 if suf1.intersection(suf2) else 0.0
        feats['legal_suffix_conflict'] = 1.0 if not suf1.intersection(suf2) else 0.0
    else:
        feats['legal_suffix_compatible'] = 0.5
        feats['legal_suffix_conflict'] = 0.0

    # --- ADDRESS FEATURES ---
    feats['addr_exact_match'] = 1.0 if r1.norm_address and r1.norm_address == r2.norm_address else 0.0
    feats['addr_compact_exact'] = 1.0 if r1.compact_address and r1.compact_address == r2.compact_address else 0.0
    feats['addr_sorted_exact'] = 1.0 if r1.sorted_address and r1.sorted_address == r2.sorted_address else 0.0

    feats['addr_ratio'] = fuzz.ratio(r1.norm_address, r2.norm_address) / 100.0 if (r1.norm_address and r2.norm_address) else 0.0
    feats['addr_token_set_ratio'] = fuzz.token_set_ratio(r1.norm_address, r2.norm_address) / 100.0 if (r1.norm_address and r2.norm_address) else 0.0
    feats['addr_partial_ratio'] = fuzz.partial_ratio(r1.norm_address, r2.norm_address) / 100.0 if (r1.norm_address and r2.norm_address) else 0.0

    aset1, aset2 = set(r1.tokens_address), set(r2.tokens_address)
    ainter = aset1.intersection(aset2)
    aunion = aset1.union(aset2)
    feats['addr_token_jaccard'] = len(ainter) / len(aunion) if aunion else 0.0
    feats['addr_token_containment'] = len(ainter) / min(len(aset1), len(aset2)) if aset1 and aset2 else 0.0

    # Address Components, Missing vs Conflict
    p1, p2 = r1.postal_code, r2.postal_code
    feats['missing_postal_flag'] = 1.0 if (not p1 or not p2) else 0.0
    if p1 and p2:
        feats['postal_exact_match'] = 1.0 if p1 == p2 else 0.0
        feats['postal_conflict'] = 1.0 if p1 != p2 else 0.0
    else:
        feats['postal_exact_match'] = 0.0
        feats['postal_conflict'] = 0.0
        
    h1, h2 = r1.house_number, r2.house_number
    feats['missing_house_flag'] = 1.0 if (not h1 or not h2) else 0.0
    if h1 and h2:
        feats['house_num_match'] = 1.0 if h1 == h2 else 0.0
        feats['house_num_conflict'] = 1.0 if h1 != h2 else 0.0
    else:
        feats['house_num_match'] = 0.0
        feats['house_num_conflict'] = 0.0

    feats['missing_address_flag'] = 1.0 if (not r1.norm_address or not r2.norm_address) else 0.0

    # Address Anchor Match
    rare_addr_inter = set(r1.rare_tokens_address).intersection(set(r2.rare_tokens_address))
    feats['addr_anchor_match'] = 1.0 if (
        (h1 and h1 == h2 and len(rare_addr_inter) >= 1) or
        (p1 and p1 == p2 and len(rare_addr_inter) >= 1) or
        (p1 and p1 == p2 and h1 and h1 == h2)
    ) else 0.0

    c1, c2 = r1.country, r2.country
    if c1 and c2:
        feats['country_match'] = 1.0 if c1 == c2 else 0.0
        feats['country_conflict'] = 1.0 if c1 != c2 else 0.0
    else:
        feats['country_match'] = 0.5
        feats['country_conflict'] = 0.0

    target_id = r2.entity_id
    feats['is_source_2'] = 1.0 if target_id.startswith('S2-') else 0.0
    feats['is_source_3'] = 1.0 if target_id.startswith('S3-') else 0.0

    feats['name_x_addr_ratio'] = feats['name_ratio'] * feats['addr_ratio']
    feats['name_x_addr_jaccard'] = feats['name_token_jaccard'] * feats['addr_token_jaccard']
    feats['cross_field_support'] = feats['name_ratio'] * (1.0 if feats['addr_token_containment'] > 0.4 or feats['addr_exact_match'] else 0.5)

    # Retrieval channel & category features (if cand_evidence passed)
    if cand_evidence:
        feats['retrieval_channel_count'] = float(cand_evidence.get('channel_count', 1))
        feats['best_retrieval_rank'] = float(cand_evidence.get('best_retrieval_rank', 1))
        feats['best_retrieval_score'] = float(cand_evidence.get('best_retrieval_score', 0.5))
        feats['is_recovery_candidate'] = float(cand_evidence.get('is_recovery', 0))
        feats['is_primary_candidate'] = float(cand_evidence.get('is_primary', 1))

        cat = cand_evidence.get('candidate_category', 'A')
        feats['cat_A'] = 1.0 if cat == 'A' else 0.0
        feats['cat_B'] = 1.0 if cat == 'B' else 0.0
        feats['cat_C'] = 1.0 if cat == 'C' else 0.0
        feats['cat_D'] = 1.0 if cat == 'D' else 0.0
        feats['cat_E'] = 1.0 if cat == 'E' else 0.0
    else:
        feats['retrieval_channel_count'] = 1.0
        feats['best_retrieval_rank'] = 1.0
        feats['best_retrieval_score'] = 0.5
        feats['is_recovery_candidate'] = 0.0
        feats['is_primary_candidate'] = 1.0
        feats['cat_A'] = 1.0
        feats['cat_B'] = 0.0
        feats['cat_C'] = 0.0
        feats['cat_D'] = 0.0
        feats['cat_E'] = 0.0

    return feats
