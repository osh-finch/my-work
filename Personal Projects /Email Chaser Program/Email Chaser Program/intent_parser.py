import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Intent:
    instruction: str
    filter_phrase: Optional[str]
    message_goal: str


def parse_user_intent(text: str) -> Intent:
    instruction = text.strip()
    filter_phrase = _extract_quoted_phrase(instruction)
    if not filter_phrase:
        filter_phrase = _extract_after_said(instruction)
    message_goal = _infer_goal(instruction)
    return Intent(instruction=instruction, filter_phrase=filter_phrase, message_goal=message_goal)


def _extract_quoted_phrase(text: str) -> Optional[str]:
    match = re.search(r"[\"“”']([^\"“”']+)[\"“”']", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_after_said(text: str) -> Optional[str]:
    lowered = text.lower()
    if "said" in lowered:
        idx = lowered.index("said") + 4
        candidate = text[idx:].strip(" :.-")
        return candidate if candidate else None
    return None


def _infer_goal(text: str) -> str:
    lowered = text.lower()
    if "final notice" in lowered:
        return "final notice"
    if "firm" in lowered or "chase" in lowered:
        return "firm chase"
    return "polite reminder"
