from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class LlmDecision:
    recipient_identifier: str
    resolved_phone: Optional[str]
    row_reference: str
    message_text: str
    personalisation_fields_used: List[str]
    confidence: float
    reason: str
    should_message: bool
    issues: List[str]


LLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "recipient_identifier": {"type": "string"},
                    "resolved_phone": {"type": ["string", "null"]},
                    "row_reference": {"type": "string"},
                    "message_text": {"type": "string"},
                    "personalisation_fields_used": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                    "should_message": {"type": "boolean"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "recipient_identifier",
                    "resolved_phone",
                    "row_reference",
                    "message_text",
                    "personalisation_fields_used",
                    "confidence",
                    "reason",
                    "should_message",
                    "issues",
                ],
            },
        }
    },
    "required": ["decisions"],
}

INFERENCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objective": {"type": "string"},
        "objective_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "recipient_column": {"type": ["string", "null"]},
        "phone_column": {"type": ["string", "null"]},
        "needs_columns": {"type": "array", "items": {"type": "string"}},
        "amount_column": {"type": ["string", "null"]},
        "personalization_columns": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "objective",
        "objective_confidence",
        "recipient_column",
        "phone_column",
        "needs_columns",
        "amount_column",
        "personalization_columns",
        "reasoning",
    ],
}


def decide_messages_with_llm(
    sheet_rows: List[Dict[str, str]],
    intent: Dict[str, str],
    column_mapping: Dict[str, str],
) -> List[LlmDecision]:
    """
    Use OpenAI API to decide who to message and draft content.
    Implementation added in Phase 3.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    system_prompt = (
        "You are an assistant that decides who should receive WhatsApp messages and drafts them. "
        "Use only the provided row data and column mapping. Do not guess phone numbers. "
        "Only include rows that should be messaged in the output. "
        "If a column has a *_formatted field, prefer that value for display. "
        "If refinement_instructions are provided, update the drafts accordingly. "
        "Return JSON that strictly matches the provided schema. Output only JSON."
    )

    payload = {
        "intent": intent,
        "column_mapping": column_mapping,
        "rows": sheet_rows,
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "message_decisions",
                "schema": LLM_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.2,
    )

    refusal = getattr(response, "refusal", None)
    if refusal:
        raise RuntimeError(f"LLM refused the request: {refusal}")

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("LLM returned no output.")

    return parse_llm_output(output_text)


def infer_sheet_context(sheet_summary: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    system_prompt = (
        "You infer the likely chase objective and relevant columns from a spreadsheet summary. "
        "Return JSON that strictly matches the schema. Output only JSON."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(sheet_summary)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "sheet_inference",
                "schema": INFERENCE_SCHEMA,
                "strict": True,
            }
        },
        temperature=0.2,
    )

    refusal = getattr(response, "refusal", None)
    if refusal:
        raise RuntimeError(f"LLM refused the request: {refusal}")

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("LLM returned no output.")

    return parse_inference_output(output_text)


def parse_llm_output(output_text: str) -> List[LlmDecision]:
    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM returned invalid JSON.") from exc

    decisions = []
    for item in data.get("decisions", []):
        decisions.append(
            LlmDecision(
                recipient_identifier=item["recipient_identifier"],
                resolved_phone=item.get("resolved_phone"),
                row_reference=item["row_reference"],
                message_text=item["message_text"],
                personalisation_fields_used=item.get("personalisation_fields_used", []),
                confidence=float(item.get("confidence", 0)),
                reason=item.get("reason", ""),
                should_message=bool(item.get("should_message", False)),
                issues=item.get("issues", []),
            )
        )

    return decisions


def parse_inference_output(output_text: str) -> Dict[str, Any]:
    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM returned invalid JSON.") from exc
    return data
