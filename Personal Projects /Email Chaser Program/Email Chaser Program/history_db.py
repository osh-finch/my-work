import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple


def history_db_init(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_e164 TEXT NOT NULL,
                sent_at_utc TEXT NOT NULL,
                message_text TEXT NOT NULL,
                sheet_id TEXT,
                row_ref TEXT,
                campaign_tag TEXT,
                status TEXT NOT NULL,
                error TEXT,
                idempotency_key TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sheet_cache (
                sheet_id TEXT PRIMARY KEY,
                tab_name TEXT,
                header_hash TEXT,
                name_col TEXT,
                decision_cols TEXT,
                updated_at_utc TEXT
            )
            """
        )
        conn.commit()


def history_last_message_for_phone(db_path: str, phone: str) -> Optional[Tuple[str, str]]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT sent_at_utc, message_text FROM message_history
            WHERE phone_e164 = ?
            ORDER BY sent_at_utc DESC
            LIMIT 1
            """,
            (phone,),
        ).fetchone()
        if not row:
            return None
        return row[0], row[1]


def history_write_sent(
    db_path: str,
    phone: str,
    message_text: str,
    sheet_id: Optional[str],
    row_ref: Optional[str],
    campaign_tag: Optional[str],
) -> None:
    _write_history(db_path, phone, message_text, sheet_id, row_ref, campaign_tag, "sent", None)


def history_write_failed(
    db_path: str,
    phone: str,
    message_text: str,
    sheet_id: Optional[str],
    row_ref: Optional[str],
    campaign_tag: Optional[str],
    error: str,
) -> None:
    _write_history(db_path, phone, message_text, sheet_id, row_ref, campaign_tag, "failed", error)


def _write_history(
    db_path: str,
    phone: str,
    message_text: str,
    sheet_id: Optional[str],
    row_ref: Optional[str],
    campaign_tag: Optional[str],
    status: str,
    error: Optional[str],
) -> None:
    sent_at = datetime.now(timezone.utc).isoformat()
    idempotency_key = _make_idempotency_key(phone, message_text, sheet_id, row_ref)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO message_history
            (phone_e164, sent_at_utc, message_text, sheet_id, row_ref, campaign_tag, status, error, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (phone, sent_at, message_text, sheet_id, row_ref, campaign_tag, status, error, idempotency_key),
        )
        conn.commit()


def _make_idempotency_key(phone: str, message_text: str, sheet_id: Optional[str], row_ref: Optional[str]) -> str:
    raw = f"{phone}|{sheet_id or ''}|{row_ref or ''}|{message_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_sheet_config(
    db_path: str,
    sheet_id: str,
    tab_name: str,
    header_hash: str,
    name_col: str,
    decision_cols_csv: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sheet_cache (sheet_id, tab_name, header_hash, name_col, decision_cols, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sheet_id) DO UPDATE SET
              tab_name=excluded.tab_name,
              header_hash=excluded.header_hash,
              name_col=excluded.name_col,
              decision_cols=excluded.decision_cols,
              updated_at_utc=excluded.updated_at_utc
            """,
            (
                sheet_id,
                tab_name,
                header_hash,
                name_col,
                decision_cols_csv,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def load_cached_sheet_config(db_path: str, sheet_id: str) -> Optional[Tuple[str, str, str, str]]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT tab_name, header_hash, name_col, decision_cols FROM sheet_cache WHERE sheet_id = ?",
            (sheet_id,),
        ).fetchone()
        if not row:
            return None
        return row[0], row[1], row[2], row[3]
