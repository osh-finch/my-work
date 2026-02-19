import llm_batch_ui


def test_optional_range_accepts_blank():
    flow = llm_batch_ui.LlmBatchChatFlow()
    assert flow.handle_input("sheet") is not None
    assert flow.handle_input("") is not None
    prompt = flow.handle_input("")
    assert prompt == "__FETCH_SUGGESTIONS__"


def test_column_confirmation_adjusts():
    flow = llm_batch_ui.LlmBatchChatFlow()
    flow.apply_suggestions(["Name", "Paid?"])
    prompt = flow.handle_input("adjust")
    assert "recipient" in prompt.lower()
