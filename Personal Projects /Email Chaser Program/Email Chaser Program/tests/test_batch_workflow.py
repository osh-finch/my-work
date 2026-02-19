import batch_workflow


def test_summarize_sheet_trims_rows():
    header = ["Name", "Paid?"]
    rows = [["Alice", "no"], ["Bob", "yes"]]
    summary = batch_workflow.summarize_sheet(header, rows, max_rows=1)
    assert summary["row_count"] == 2
    assert len(summary["sample_rows"]) == 1


def test_format_currency():
    assert batch_workflow.format_currency("1000") == "£1,000.00"
    assert batch_workflow.format_currency("1000.5") == "£1,000.50"
