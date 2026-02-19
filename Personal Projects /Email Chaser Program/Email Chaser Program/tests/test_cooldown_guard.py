from datetime import datetime, timedelta, timezone

from cooldown_guard import apply_cooldown_guard
from plan_model import PlanRecipient


def test_cooldown_guard_excludes_recent():
    recent = datetime.now(timezone.utc) - timedelta(days=2)
    old = datetime.now(timezone.utc) - timedelta(days=10)
    recipients = [
        PlanRecipient(row_index=2, full_name="Alice", phone="+1", message_text="x", last_messaged_at=recent.isoformat()),
        PlanRecipient(row_index=3, full_name="Bob", phone="+2", message_text="x", last_messaged_at=old.isoformat()),
    ]
    included, excluded = apply_cooldown_guard(recipients, cooldown_days=7, include_recent=False)
    assert len(included) == 1
    assert len(excluded) == 1
    assert excluded[0].full_name == "Alice"
