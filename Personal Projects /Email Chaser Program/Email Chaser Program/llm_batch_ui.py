from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BatchFlowState:
    sheet_url_or_id: Optional[str] = None
    worksheet_name: Optional[str] = None
    value_range: Optional[str] = None
    recipient_column: Optional[str] = None
    phone_column: Optional[str] = None
    needs_columns: Optional[List[str]] = None
    amount_column: Optional[str] = None
    personalization_columns: Optional[List[str]] = None
    goal_tone: Optional[str] = None
    confirmation_mode: str = "each"
    objective: Optional[str] = None
    objective_confidence: float = 0.0
    refinement_notes: Optional[str] = None
    last_header: Optional[List[str]] = None


class LlmBatchChatFlow:
    def __init__(self):
        self.state = BatchFlowState()
        self._steps = [
            ("sheet_url_or_id", "Which Google Sheet URL or ID should I read?"),
            ("worksheet_name", "Which worksheet/tab name should I use? (Press Enter to use the first tab)", True),
            ("value_range", "Optional range (e.g., A1:D200). Leave blank for full sheet.", True),
        ]
        self._step_index = 0
        self._awaiting_suggestions = False
        self._post_mapping_stage = False
        self._awaiting_column_confirmation = False
        self._ready_for_refinement = False
        self._awaiting_freeform_config = False
        self._missing_items: List[str] = []

    def start_prompt(self) -> str:
        return self._steps[0][1]

    def allows_empty_input(self) -> bool:
        if self._awaiting_column_confirmation:
            return True
        if self._post_mapping_stage:
            return True
        if self._step_index < len(self._steps):
            _, _, *optional = self._steps[self._step_index]
            return bool(optional and optional[0])
        return False

    def handle_input(self, text: str) -> Optional[str]:
        if self._ready_for_refinement:
            self.state.refinement_notes = text.strip()
            return "__REFINE_DRAFTS__"

        if self._awaiting_suggestions:
            return "__FETCH_SUGGESTIONS__"

        if self._awaiting_column_confirmation:
            return self._handle_column_confirmation(text)

        if self._post_mapping_stage:
            return self._handle_post_mapping_input(text)

        key, prompt, *optional = self._steps[self._step_index]
        is_optional = bool(optional and optional[0])

        if not text and not is_optional:
            return "Please provide a value."

        if key == "needs_columns":
            value = [part.strip() for part in text.split(",") if part.strip()]
            if not value:
                return "Please provide at least one column."
            self.state.needs_columns = value
        elif key == "personalization_columns":
            value = [part.strip() for part in text.split(",") if part.strip()]
            self.state.personalization_columns = value
        elif key == "confirmation_mode":
            mode = text.strip().lower()
            if not mode:
                mode = "each"
            if mode not in {"each", "batch", "auto"}:
                return "Please choose: each, batch, or auto."
            self.state.confirmation_mode = mode
        elif key == "value_range":
            cleaned = text.strip()
            if not cleaned or cleaned.lower() in {"blank", "none", "n/a"}:
                self.state.value_range = None
            else:
                self.state.value_range = cleaned
        elif key == "worksheet_name":
            cleaned = text.strip()
            self.state.worksheet_name = cleaned or None
        elif key == "phone_column":
            self.state.phone_column = text.strip() or None
        elif key == "amount_column":
            self.state.amount_column = text.strip() or None
        else:
            setattr(self.state, key, text.strip())

        self._step_index += 1
        if self._step_index >= len(self._steps):
            self._awaiting_suggestions = True
            return "__FETCH_SUGGESTIONS__"
        return self._steps[self._step_index][1]

    def apply_suggestions(self, header: List[str]) -> str:
        self.state.last_header = header
        suggestions = self._infer_suggestions(header)
        if not self.state.recipient_column:
            self.state.recipient_column = suggestions.get("recipient_column")
        if not self.state.phone_column:
            self.state.phone_column = suggestions.get("phone_column")
        if not self.state.needs_columns:
            self.state.needs_columns = suggestions.get("needs_columns")
        if not self.state.amount_column:
            self.state.amount_column = suggestions.get("amount_column")
        if not self.state.personalization_columns:
            self.state.personalization_columns = suggestions.get("personalization_columns")

        self._awaiting_suggestions = False
        self._awaiting_column_confirmation = True
        summary = self._format_suggestion_summary()
        return (
            "Here’s my plan based on the sheet:\n"
            f"{summary}\n"
            "Is this correct? Press Enter to confirm, or type adjustments in one sentence."
        )

    def apply_inference(self, header: List[str], inference: Optional[Dict[str, object]]) -> str:
        if inference:
            self.state.objective = inference.get("objective") or self.state.objective
            self.state.objective_confidence = float(inference.get("objective_confidence", 0))
            self.state.recipient_column = inference.get("recipient_column")
            self.state.phone_column = inference.get("phone_column")
            self.state.needs_columns = inference.get("needs_columns")
            self.state.amount_column = inference.get("amount_column")
            self.state.personalization_columns = inference.get("personalization_columns")
        return self.apply_suggestions(header)

    def _handle_column_confirmation(self, text: str) -> Optional[str]:
        choice = text.strip().lower()
        if choice == "adjust":
            self._awaiting_freeform_config = True
            return self._config_prompt()
        if choice:
            updated = self._apply_column_adjustments(choice)
            if updated:
                return (
                    "Got it — I’ve updated the column choices. Here’s the new plan:\n"
                    f"{self._format_suggestion_summary()}\n"
                    "Is this correct now? (Enter to accept, or type 'adjust' to change columns)"
                )
            return (
                "I didn’t catch a specific column change. "
                "If you want to adjust, say something like “use Paid? for chasing” or type ‘adjust’."
            )
        self._awaiting_column_confirmation = False
        if not self.state.recipient_column or not self.state.needs_columns:
            self._awaiting_freeform_config = True
            return self._config_prompt()
        return self._post_mapping_prompt()

    def _post_mapping_prompt(self) -> str:
        self._post_mapping_stage = True
        if not self.state.goal_tone:
            return "What tone should I use, and how should I confirm? (each/batch/auto) [polite reminder, each]"
        return "How should I confirm? (each/batch/auto)"

    def _handle_post_mapping_input(self, text: str) -> Optional[str]:
        if not self.state.goal_tone:
            cleaned = text.strip().lower()
            if not cleaned:
                self.state.goal_tone = "polite reminder"
                self.state.confirmation_mode = "each"
                self._post_mapping_stage = False
                return None
            if "firm" in cleaned:
                self.state.goal_tone = "firm chase"
            elif "final" in cleaned:
                self.state.goal_tone = "final notice"
            else:
                self.state.goal_tone = cleaned

            if "auto" in cleaned:
                self.state.confirmation_mode = "auto"
            elif "batch" in cleaned:
                self.state.confirmation_mode = "batch"
            else:
                self.state.confirmation_mode = "each"

            self._post_mapping_stage = False
            return None

        mode = text.strip().lower()
        if not mode:
            mode = "each"
        if mode not in {"each", "batch", "auto"}:
            return "Please choose: each, batch, or auto."
        self.state.confirmation_mode = mode
        self._post_mapping_stage = False
        return None

    def enable_refinement_mode(self):
        self._ready_for_refinement = True

    def disable_refinement_mode(self):
        self._ready_for_refinement = False

    def start_manual_override(self):
        self._awaiting_column_confirmation = False
        self._awaiting_freeform_config = True

    def restart_worksheet_prompt(self):
        self._awaiting_suggestions = False
        self._post_mapping_stage = False
        self._awaiting_column_confirmation = False
        self._awaiting_freeform_config = False
        self._step_index = 1

    def handle_freeform_config(self, text: str) -> Optional[str]:
        self._awaiting_freeform_config = False
        self._apply_freeform_config(text)
        missing = self._missing_required()
        if missing:
            self._awaiting_freeform_config = True
            return self._config_prompt(missing)
        return self._post_mapping_prompt()

    def _config_prompt(self, missing: Optional[List[str]] = None) -> str:
        if not missing:
            return (
                "Tell me in one message: which column has the recipient name/ID, "
                "which column shows who needs chasing, and (optionally) which column has phone numbers and amounts."
            )
        missing_text = ", ".join(missing)
        return f"I still need: {missing_text}. Please answer in one message."

    def _missing_required(self) -> List[str]:
        missing = []
        if not self.state.recipient_column:
            missing.append("recipient column")
        if not self.state.needs_columns:
            missing.append("needs-chasing column(s)")
        return missing

    def _apply_freeform_config(self, text: str) -> None:
        lowered = text.lower()
        header = self.state.last_header or []
        header_lower = [h.lower() for h in header]

        def match_header(target: str) -> Optional[str]:
            if target in header_lower:
                return header[header_lower.index(target)]
            return None

        if "paid" in lowered:
            candidate = match_header("paid?") or match_header("paid")
            if candidate:
                self.state.needs_columns = [candidate]
        if "to pay" in lowered or "amount" in lowered:
            candidate = match_header("to pay") or match_header("amount") or match_header("amount owed")
            if candidate:
                self.state.amount_column = candidate
        if "phone" in lowered or "mobile" in lowered:
            candidate = match_header("phone") or match_header("phone number") or match_header("mobile")
            if candidate:
                self.state.phone_column = candidate
        if "name" in lowered or "recipient" in lowered:
            candidate = match_header("name") or match_header("full name") or match_header("member")
            if candidate:
                self.state.recipient_column = candidate

    def _infer_suggestions(self, header: List[str]) -> Dict[str, object]:
        lower = [h.lower() for h in header]

        def find_first(candidates):
            for cand in candidates:
                if cand in lower:
                    return header[lower.index(cand)]
            return None

        recipient = find_first(["name", "full name", "member", "recipient"])
        phone = find_first(["phone", "phone number", "mobile", "number", "whatsapp"])
        needs = []
        for cand in ["paid?", "paid", "status", "due", "owed", "balance"]:
            if cand in lower:
                needs.append(header[lower.index(cand)])
                break
        amount = find_first(["to pay", "amount", "amount owed", "balance"])

        personalization = []
        for cand in ["email", "team", "role"]:
            if cand in lower:
                personalization.append(header[lower.index(cand)])

        return {
            "recipient_column": recipient,
            "phone_column": phone,
            "needs_columns": needs,
            "amount_column": amount,
            "personalization_columns": personalization,
        }

    def _format_suggestion_summary(self) -> str:
        parts = []
        recipient = self.state.recipient_column or "no recipient column detected"
        parts.append(f"Recipients look like they’re in the “{recipient}” column.")

        if self.state.phone_column:
            parts.append(f"I’ll use phone numbers from “{self.state.phone_column}”.")
        else:
            parts.append("I didn’t detect a phone column; I’ll try to match names from your contacts list.")

        needs = ", ".join(self.state.needs_columns or []) or "no status columns detected"
        parts.append(f"I’ll decide who needs chasing using “{needs}”.")

        if self.state.amount_column:
            parts.append(f"Amounts will come from “{self.state.amount_column}”.")

        if self.state.personalization_columns:
            parts.append(
                f"I’ll personalize using {', '.join(self.state.personalization_columns)}."
            )

        return "\n".join(f"- {part}" for part in parts)

    def _apply_column_adjustments(self, text: str) -> bool:
        lowered = text.lower()
        updated = False

        def remove_if_present(name: Optional[str]) -> Optional[str]:
            nonlocal updated
            if name and name.lower() in lowered:
                updated = True
                return None
            return name

        if self.state.personalization_columns:
            remaining = []
            for col in self.state.personalization_columns:
                if col.lower() in lowered:
                    updated = True
                    continue
                remaining.append(col)
            self.state.personalization_columns = remaining

        if self.state.needs_columns:
            remaining = []
            for col in self.state.needs_columns:
                if col.lower() in lowered:
                    updated = True
                    continue
                remaining.append(col)
            self.state.needs_columns = remaining

        if "paid" in lowered:
            self.state.needs_columns = ["Paid?"]
            updated = True
        if "to pay" in lowered:
            self.state.amount_column = "To Pay"
            updated = True
        if "personal" in lowered or "personalisation" in lowered or "personalization" in lowered:
            self.state.personalization_columns = []
            updated = True

        self.state.amount_column = remove_if_present(self.state.amount_column)
        self.state.phone_column = remove_if_present(self.state.phone_column)
        self.state.recipient_column = remove_if_present(self.state.recipient_column)

        return updated
