import os
import tempfile

import history_db


def test_history_write_and_last_message():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "history.db")
        history_db.history_db_init(db_path)
        history_db.history_write_sent(
            db_path,
            "+447700900123",
            "Test message",
            "sheet1",
            "row 2",
            "campaign",
        )
        last = history_db.history_last_message_for_phone(db_path, "+447700900123")
        assert last is not None
        sent_at, message_text = last
        assert "Test message" in message_text
