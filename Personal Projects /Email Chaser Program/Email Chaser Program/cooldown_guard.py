from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from plan_model import PlanRecipient


def apply_cooldown_guard(
    recipients: List[PlanRecipient],
    cooldown_days: int,
    include_recent: bool,
) -> Tuple[List[PlanRecipient], List[PlanRecipient]]:
    if include_recent:
        return recipients, []

    cutoff = datetime.now(timezone.utc) - timedelta(days=cooldown_days)
    included = []
    excluded = []
    for recipient in recipients:
        if not recipient.last_messaged_at:
            included.append(recipient)
            continue
        try:
            last_dt = datetime.fromisoformat(recipient.last_messaged_at)
        except ValueError:
            included.append(recipient)
            continue
        if last_dt >= cutoff:
            recipient.excluded_due_to_cooldown = True
            excluded.append(recipient)
        else:
            included.append(recipient)
    return included, excluded
