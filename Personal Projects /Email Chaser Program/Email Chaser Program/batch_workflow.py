from typing import Callable, Dict, List, Optional, Tuple

import contact_resolution
import llm_decider


def summarize_sheet(header: List[str], rows: List[List[str]], max_rows: int = 20) -> Dict[str, object]:
    sample_rows = []
    for row in rows[:max_rows]:
        row_dict = dict(zip(header, row))
        trimmed = {k: _trim_value(v) for k, v in row_dict.items()}
        sample_rows.append(trimmed)

    return {
        "columns": header,
        "row_count": len(rows),
        "sample_rows": sample_rows,
    }


def build_rows_payload(
    header: List[str],
    rows: List[List[str]],
    row_numbers: List[int],
    recipient_column: str,
    phone_column: Optional[str],
    needs_columns: List[str],
    amount_column: Optional[str],
    personalization_columns: List[str],
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    rows_payload = []
    row_lookup: Dict[str, Dict[str, str]] = {}
    for row_values, row_number in zip(rows, row_numbers):
        row_dict = dict(zip(header, row_values))
        payload = {
            "row_reference": f"row {row_number}",
            recipient_column: row_dict.get(recipient_column, ""),
        }

        if phone_column:
            payload[phone_column] = row_dict.get(phone_column, "")

        for col in needs_columns:
            payload[col] = row_dict.get(col, "")

        for col in personalization_columns:
            payload[col] = row_dict.get(col, "")

        if amount_column:
            payload[amount_column] = row_dict.get(amount_column, "")
            payload[f"{amount_column}_formatted"] = format_currency(row_dict.get(amount_column, ""))

        rows_payload.append(payload)
        row_lookup[payload["row_reference"]] = row_dict

    return rows_payload, row_lookup


def apply_contact_resolution(
    decisions: List[llm_decider.LlmDecision],
    row_lookup: Dict[str, Dict[str, str]],
    phone_column: Optional[str],
    contacts_df,
    log_fn: Callable[[str], None],
) -> List[llm_decider.LlmDecision]:
    enriched = []
    for decision in decisions:
        row = row_lookup.get(decision.row_reference, {})
        if not row:
            decision.issues.append("unknown_row_reference")
            log_fn(f"Row reference not found for {decision.recipient_identifier}: {decision.row_reference}.")
        row_phone = row.get(phone_column) if phone_column else None
        resolution = contact_resolution.resolve_contact(
            decision.recipient_identifier,
            row_phone,
            contacts_df,
        )
        decision.resolved_phone = resolution.resolved_phone
        if resolution.issue:
            decision.issues.append(resolution.issue)
            log_fn(
                f"Skipping {decision.recipient_identifier}: {resolution.issue} (row {decision.row_reference})."
            )
        enriched.append(decision)
    return enriched


def format_currency(raw_value: str, symbol: str = "£") -> str:
    if raw_value is None:
        return ""
    raw = str(raw_value).strip().replace(",", "")
    if not raw:
        return ""
    try:
        amount = float(raw)
    except ValueError:
        return str(raw_value)
    return f"{symbol}{amount:,.2f}"


def _trim_value(value: object, limit: int = 120) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
