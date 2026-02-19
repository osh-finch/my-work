import json

import llm_decider


def test_parse_llm_output_happy_path():
    payload = {
        "decisions": [
            {
                "recipient_identifier": "Alice",
                "resolved_phone": "+447700900123",
                "row_reference": "row 2",
                "message_text": "Hi Alice",
                "personalisation_fields_used": ["Name"],
                "confidence": 0.9,
                "reason": "Unpaid",
                "should_message": True,
                "issues": [],
            }
        ]
    }
    decisions = llm_decider.parse_llm_output(json.dumps(payload))
    assert len(decisions) == 1
    assert decisions[0].recipient_identifier == "Alice"


def test_parse_llm_output_invalid_json():
    try:
        llm_decider.parse_llm_output("{not-json}")
    except RuntimeError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for invalid JSON")


def test_parse_inference_output():
    payload = {
        "objective": "payment reminder",
        "objective_confidence": 0.8,
        "recipient_column": "Name",
        "phone_column": None,
        "needs_columns": ["Paid?"],
        "amount_column": "To Pay",
        "personalization_columns": [],
        "reasoning": "Based on Paid? and To Pay columns.",
    }
    data = llm_decider.parse_inference_output(json.dumps(payload))
    assert data["objective"] == "payment reminder"
