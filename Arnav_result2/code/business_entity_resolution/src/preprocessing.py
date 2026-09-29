"""
ML Challenge 2026 - Business Entity Resolution Pipeline
Enhanced Preprocessing & Text Normalization Module

Handles:
- Corporate legal suffix normalization and stripping
- Address abbreviation expansion and standardization
- URL / domain name cleaning (e.g. 'xyz.com' -> 'xyz')
- Alphanumeric slug generation for cross-format matching
- Robust extraction of postal PIN/ZIP codes and building numbers
"""

import re
import unicodedata
from typing import Optional
import pandas as pd


# Corporate suffix pattern (ordered by length descending to match composite phrases first)
LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b("
    r"private limited|pvt ltd|pvt limited|private ltd|"
    r"incorporated|corporation|enterprise|"
    r"limited|pvt|ltd|llc|inc|corp|co|llp|pc|plc|gmbh|sarl|sa"
    r")\b",
    re.IGNORECASE,
)

# Address abbreviation mappings
ADDRESS_ABBR_MAP = {
    r"\bst\b|\bstr\b": "street",
    r"\brd\b": "road",
    r"\bave\b|\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b|\bdrv\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b|\bapartment\b": "unit",
    r"\bbldg\b": "building",
    r"\bfl\b|\bflr\b": "floor",
    r"\bopp\b|\bopposite\b": "opp",
    r"\bnr\b|\bnear\b": "near",
    r"\bh\.?no\.?\b|\bhno\b": "house",
    r"\bno\.?\b": "number",
}

ADDRESS_COMPILED_ABBRS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR_MAP.items()]

# URL domain cleaning pattern
URL_PREFIX_PATTERN = re.compile(r"^(https?:\/\/)?(www\.)?", re.IGNORECASE)
URL_SUFFIX_PATTERN = re.compile(r"\.(com|in|org|net|co\.in|co|io|fr|gov|edu)(\/.*)?$", re.IGNORECASE)

POSTAL_CODE_PATTERN = re.compile(r"\b(\d{5,6})\b")
HOUSE_NUMBER_PATTERN = re.compile(r"^\s*(\d+[a-zA-Z]?)", re.IGNORECASE)


def clean_base_text(text: Optional[str]) -> str:
    """Standard unicode normalization, lowercasing, and symbol conversion."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).lower()
    text = text.replace("&", " and ").replace("@", " at ").replace("+", " plus ")
    text = text.replace("/", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_url_slug(text: Optional[str]) -> str:
    """Extract clean domain slug from URL or text."""
    if not text:
        return ""
    t = str(text).strip().lower()
    t = URL_PREFIX_PATTERN.sub("", t)
    t = URL_SUFFIX_PATTERN.sub("", t)
    return re.sub(r"[^\w]", "", t)


def normalize_business_name(name: Optional[str]) -> str:
    """Normalize business entity name, stripping corporate suffixes safely."""
    cleaned = clean_base_text(name)
    if not cleaned:
        return ""
    stripped = LEGAL_SUFFIXES_PATTERN.sub("", cleaned)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped if len(stripped) >= 2 else cleaned


def normalize_address(address: Optional[str]) -> str:
    """Normalize address and expand standard abbreviations."""
    cleaned = clean_base_text(address)
    if not cleaned:
        return ""
    for pattern, repl in ADDRESS_COMPILED_ABBRS:
        cleaned = pattern.sub(repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_country(country: Optional[str]) -> str:
    """Normalize country strings (US, India, France, etc.)."""
    if country is None or pd.isna(country):
        return ""
    c = str(country).lower().strip()
    if c in ("us", "usa", "united states", "united states of america"):
        return "us"
    if c in ("in", "ind", "india"):
        return "india"
    if c in ("fr", "fra", "france"):
        return "france"
    return c


def extract_postal_code(address: Optional[str]) -> str:
    """Extract 5 or 6 digit postal code."""
    if not address or pd.isna(address):
        return ""
    m = POSTAL_CODE_PATTERN.search(str(address))
    return m.group(1) if m else ""


def extract_house_number(address: Optional[str]) -> str:
    """Extract leading house/building number."""
    if not address or pd.isna(address):
        return ""
    m = HOUSE_NUMBER_PATTERN.match(str(address))
    return m.group(1).lower() if m else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to DataFrame without modifying original columns.
    Adds: name_norm, address_norm, country_norm, name_slug, postal_code, house_number.
    """
    df = df.copy()

    if "business_name" in df.columns:
        df["name_norm"] = df["business_name"].apply(normalize_business_name)
        df["name_slug"] = df["business_name"].apply(clean_url_slug)
    else:
        df["name_norm"] = ""
        df["name_slug"] = ""

    if "business_address" in df.columns:
        df["address_norm"] = df["business_address"].apply(normalize_address)
        df["postal_code"] = df["business_address"].apply(extract_postal_code)
        df["house_number"] = df["business_address"].apply(extract_house_number)
    else:
        df["address_norm"] = ""
        df["postal_code"] = ""
        df["house_number"] = ""

    if "country" in df.columns:
        df["country_norm"] = df["country"].apply(normalize_country)
    else:
        df["country_norm"] = ""

    return df
