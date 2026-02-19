from dataclasses import dataclass
from typing import List, Optional, Tuple


DECISION_KEYWORDS = [
    "decision",
    "status",
    "accommodation",
    "outbound",
    "returning",
    "unsure",
    "selection",
    "paid",
    "owed",
]

NAME_KEYWORDS = ["full name", "name", "recipient", "player", "member"]


@dataclass
class SheetConfig:
    tab_name: str
    name_col: str
    decision_cols: List[str]
    confidence: float


def infer_sheet_config(
    header: List[str],
    rows: List[List[str]],
    tab_name: str,
    target_phrase: Optional[str],
) -> SheetConfig:
    name_col = _infer_name_col(header, rows)
    decision_cols = _infer_decision_cols(header, rows, target_phrase)
    confidence = 0.0
    if name_col:
        confidence += 0.5
    if decision_cols:
        confidence += 0.5
    return SheetConfig(tab_name=tab_name, name_col=name_col, decision_cols=decision_cols, confidence=confidence)


def _infer_name_col(header: List[str], rows: List[List[str]]) -> str:
    lower = [h.lower() for h in header]
    for kw in NAME_KEYWORDS:
        if kw in lower:
            return header[lower.index(kw)]

    best_idx = 0
    best_score = -1
    for idx, col in enumerate(header):
        score = 0
        for row in rows[:20]:
            if idx >= len(row):
                continue
            val = str(row[idx]).strip()
            if not val:
                continue
            if _looks_like_name(val):
                score += 1
        if score > best_score:
            best_score = score
            best_idx = idx
    return header[best_idx] if header else ""


def _infer_decision_cols(
    header: List[str],
    rows: List[List[str]],
    target_phrase: Optional[str],
) -> List[str]:
    lower = [h.lower() for h in header]
    cols = []
    for idx, col in enumerate(lower):
        if any(kw in col for kw in DECISION_KEYWORDS):
            cols.append(header[idx])
            continue
        if target_phrase:
            for row in rows[:50]:
                if idx >= len(row):
                    continue
                if target_phrase.lower() in str(row[idx]).lower():
                    cols.append(header[idx])
                    break
    return list(dict.fromkeys(cols))


def _looks_like_name(value: str) -> bool:
    if any(char.isdigit() for char in value):
        return False
    parts = value.split()
    return 1 <= len(parts) <= 4
