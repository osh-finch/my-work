from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gspread
from oauth2client.service_account import ServiceAccountCredentials


SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


@dataclass
class SheetData:
    header: List[str]
    rows: List[List[str]]
    row_numbers: List[int]
    source_sheet: str
    source_range: Optional[str]

    def as_dict_rows(self) -> List[Dict[str, str]]:
        return [dict(zip(self.header, row)) for row in self.rows]


def fetch_sheet_rows(
    sheet_url_or_id: str,
    worksheet_name: Optional[str],
    value_range: Optional[str] = None,
) -> SheetData:
    """
    Fetch rows from Google Sheets. Implementation added in Phase 2.
    """
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", SCOPE)
    client = gspread.authorize(creds)

    if "http" in sheet_url_or_id:
        spreadsheet = client.open_by_url(sheet_url_or_id)
    else:
        spreadsheet = client.open_by_key(sheet_url_or_id)

    worksheet = _resolve_worksheet(spreadsheet, worksheet_name)

    if value_range:
        values = worksheet.get(value_range)
        start_row = _parse_range_start_row(value_range)
    else:
        values = worksheet.get_all_values()
        start_row = 1

    if not values:
        return SheetData(
            header=[],
            rows=[],
            row_numbers=[],
            source_sheet=worksheet.title,
            source_range=value_range,
        )

    header = values[0]
    rows = values[1:]
    row_numbers = list(range(start_row + 1, start_row + 1 + len(rows)))
    return SheetData(
        header=header,
        rows=rows,
        row_numbers=row_numbers,
        source_sheet=worksheet.title,
        source_range=value_range,
    )


def _resolve_worksheet(spreadsheet, worksheet_name: Optional[str]):
    worksheets = spreadsheet.worksheets()
    if not worksheet_name:
        return worksheets[0]

    try:
        return spreadsheet.worksheet(worksheet_name)
    except Exception:
        lower = worksheet_name.strip().lower()
        for sheet in worksheets:
            if sheet.title.strip().lower() == lower:
                return sheet
        available = ", ".join([s.title for s in worksheets])
        raise ValueError(f"Worksheet not found: {worksheet_name}. Available tabs: {available}")


def _parse_range_start_row(a1_range: str) -> int:
    _, _, row = _parse_a1_range(a1_range)
    return row if row is not None else 1


def _parse_a1_range(a1_range: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse a minimal A1 range like 'Sheet1!A2:D100' or 'A2:D100'.
    Returns (sheet, column_start, row_start).
    """
    if "!" in a1_range:
        sheet_part, cell_part = a1_range.split("!", 1)
    else:
        sheet_part, cell_part = None, a1_range

    col = ""
    row_digits = ""
    for ch in cell_part:
        if ch.isalpha() and not row_digits:
            col += ch.upper()
        elif ch.isdigit():
            row_digits += ch
        elif ch == ":":
            break

    row_start = int(row_digits) if row_digits else None
    return sheet_part, col or None, row_start
