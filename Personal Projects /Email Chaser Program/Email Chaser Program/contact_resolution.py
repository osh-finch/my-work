from dataclasses import dataclass
from typing import Optional
import math

import pandas as pd

import chaser_logic


@dataclass
class ContactResolution:
    recipient_identifier: str
    resolved_phone: Optional[str]
    source: str
    issue: Optional[str]


def normalize_phone(raw_phone: str) -> Optional[str]:
    """
    Normalize to E.164-like string when possible. Implementation added in Phase 2.
    """
    if raw_phone is None:
        return None

    if isinstance(raw_phone, float):
        if math.isnan(raw_phone):
            return None
        if raw_phone.is_integer():
            raw_phone = str(int(raw_phone))
        else:
            return None
    elif isinstance(raw_phone, int):
        raw_phone = str(raw_phone)
    elif not isinstance(raw_phone, str):
        raw_phone = str(raw_phone)

    raw_phone = raw_phone.strip()
    if "e" in raw_phone.lower():
        try:
            float_val = float(raw_phone)
        except ValueError:
            return None
        if math.isnan(float_val) or not float_val.is_integer():
            return None
        raw_phone = str(int(float_val))
    if raw_phone.endswith(".0") and raw_phone[:-2].isdigit():
        raw_phone = raw_phone[:-2]
    if not raw_phone:
        return None

    cleaned = []
    for ch in raw_phone:
        if ch.isdigit():
            cleaned.append(ch)
        elif ch == "+" and not cleaned:
            cleaned.append(ch)
        elif ch in " ()-.\t":
            continue
        else:
            return None

    normalized = "".join(cleaned)
    if normalized.startswith("00"):
        normalized = "+" + normalized[2:]

    if normalized.startswith("+"):
        digits = normalized[1:]
    else:
        digits = normalized
        normalized = "+" + digits

    if not digits.isdigit():
        return None

    if len(digits) < 8 or len(digits) > 15:
        return None

    return normalized


def resolve_contact(
    recipient_identifier: str,
    row_phone: Optional[str],
    contacts_df,
) -> ContactResolution:
    """
    Resolve contact using sheet phone or contacts store. Implementation added in Phase 2.
    """
    normalized_sheet_phone = normalize_phone(row_phone) if row_phone else None
    if normalized_sheet_phone:
        return ContactResolution(
            recipient_identifier=recipient_identifier,
            resolved_phone=normalized_sheet_phone,
            source="sheet",
            issue=None,
        )

    if contacts_df is None or not isinstance(contacts_df, pd.DataFrame):
        return ContactResolution(
            recipient_identifier=recipient_identifier,
            resolved_phone=None,
            source="contacts",
            issue="missing_contact_store",
        )

    name_col = (
        chaser_logic.CONTACTS_NAME_COLUMN
        if chaser_logic.CONTACTS_NAME_COLUMN in contacts_df.columns
        else contacts_df.columns[0]
    )
    phone_col = (
        chaser_logic.CONTACTS_NUMBER_COLUMN
        if chaser_logic.CONTACTS_NUMBER_COLUMN in contacts_df.columns
        else contacts_df.columns[1]
    )

    matches = contacts_df[contacts_df[name_col].str.lower() == recipient_identifier.lower()]
    if matches.empty:
        return ContactResolution(
            recipient_identifier=recipient_identifier,
            resolved_phone=None,
            source="contacts",
            issue="missing_contact",
        )

    for _, row in matches.iterrows():
        normalized = normalize_phone(row.get(phone_col))
        if normalized:
            return ContactResolution(
                recipient_identifier=recipient_identifier,
                resolved_phone=normalized,
                source="contacts",
                issue=None,
            )

    return ContactResolution(
        recipient_identifier=recipient_identifier,
        resolved_phone=None,
        source="contacts",
        issue="invalid_phone",
    )
