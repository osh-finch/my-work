from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class PlanRecipient:
    row_index: int
    full_name: str
    phone: Optional[str]
    message_text: str
    unresolved_reason: Optional[str] = None
    last_messaged_at: Optional[str] = None
    last_messaged_relative: Optional[str] = None
    last_message_snippet: Optional[str] = None
    excluded_due_to_cooldown: bool = False


@dataclass
class Plan:
    sheet_id: str
    sheet_url: str
    tab_name: str
    value_range: Optional[str]
    name_col: str
    decision_cols: List[str]
    filter_description: str
    filter_predicate: Callable[[dict], bool]
    recipients: List[PlanRecipient] = field(default_factory=list)
    unresolved: List[PlanRecipient] = field(default_factory=list)
    excluded_recent: List[PlanRecipient] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    objective: Optional[str] = None
    instruction: Optional[str] = None
