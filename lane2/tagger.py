"""Deterministic typed-value tagger (shared by the §2.7 screen's C2, the
FactLedger detectors of Step 4, and result-fact extraction).

Types: MONEY, DATE, TIME, DURATION, PERCENT, QUANTITY, IDENT.
Each hit: {type, value (normalized), surface, start, end}.
Number words are normalized for the numeric types (Step 4 normalizer).
Hedged rounding ("about fifty dollars") still yields a typed value; the
screen's C2 hedge rule consumes the `hedged` flag.
"""
import re

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}

_MONTHS = ("january february march april may june july august september "
           "october november december jan feb mar apr jun jul aug sep sept "
           "oct nov dec").split()

_HEDGE = re.compile(r"\b(about|around|roughly|approximately|nearly|almost|"
                    r"just under|just over|~)\s*$", re.I)


def parse_number_words(s):
    """Parse 'one thousand two hundred' / 'forty two' -> number or None."""
    toks = s.lower().replace("-", " ").split()
    if not toks or any(t not in NUMBER_WORDS and t != "and" for t in toks):
        return None
    total, cur = 0, 0
    seen = False
    for t in toks:
        if t == "and":
            continue
        v = NUMBER_WORDS[t]
        seen = True
        if v == 100:
            cur = (cur or 1) * 100
        elif v >= 1000:
            total += (cur or 1) * v
            cur = 0
        else:
            cur += v
    return (total + cur) if seen else None


_NUM = r"(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](?:one|two|three|four|five|six|seven|eight|nine))"

_PATTERNS = [
    ("MONEY", re.compile(
        r"\$\s?\d[\d,]*(?:\.\d{2})?\b|\b" + _NUM + r"\s+(?:dollars?|bucks|usd)\b",
        re.I)),
    ("DATE", re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:" +
        "|".join(_MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b",
        re.I)),
    ("TIME", re.compile(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b|"
        r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b|\bnoon\b|\bmidnight\b", re.I)),
    ("PERCENT", re.compile(
        r"\b\d+(?:\.\d+)?\s?%|\b" + _NUM + r"\s+percent\b", re.I)),
    ("DURATION", re.compile(
        r"\b" + _NUM + r"\s+(?:seconds?|minutes?|hours?|days?|weeks?|"
        r"months?|years?)\b", re.I)),
    ("QUANTITY", re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s+(?:items?|files?|emails?|messages?|pages?|"
        r"people|tickets?|photos?|songs?|calls?|meetings?|tasks?|jobs?|"
        r"entries|records?|results?|inches|inch|feet|foot|ft|miles?|"
        r"pounds?|lbs?|kg|grams?|liters?|gallons?|acres?|yards?|"
        r"meters?|cm|mm)\b", re.I)),
    ("IDENT", re.compile(
        r"\b[A-Z]{2,}-\d{3,}\b|\b\d{9,}\b|"
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
        r"#\d{3,}\b")),
]


def _normalize(typ, surface):
    s = surface.strip()
    if typ == "MONEY":
        m = re.search(r"[\d][\d,]*(?:\.\d+)?", s)
        if m:
            return m.group(0).replace(",", "")
        v = parse_number_words(s)
        return str(v) if v is not None else s.lower()
    if typ in ("PERCENT", "DURATION", "QUANTITY"):
        m = re.search(r"[\d][\d,]*(?:\.\d+)?", s)
        if m:
            return m.group(0).replace(",", "")
        v = parse_number_words(s)
        return str(v) if v is not None else s.lower()
    if typ == "DATE":
        return re.sub(r"\s+", " ", s.lower())
    if typ == "TIME":
        return re.sub(r"\s+", "", s.lower()).replace(".", "")
    return s


def extract_typed_values(text):
    """All typed spans, non-overlapping, in text order."""
    hits = []
    spans = []
    for typ, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(not (e <= a or s >= b) for a, b in spans):
                continue
            spans.append((s, e))
            hedged = bool(_HEDGE.search(text[:s]))
            hits.append({"type": typ, "value": _normalize(typ, m.group(0)),
                         "surface": m.group(0), "start": s, "end": e,
                         "hedged": hedged})
    hits.sort(key=lambda h: h["start"])
    return hits


def typed_multiset(text):
    return sorted((h["type"], h["value"]) for h in extract_typed_values(text))
