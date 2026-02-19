import sheet_inference


def test_infer_name_and_decision_cols():
    header = ["Full Name", "Paid?", "Notes"]
    rows = [
        ["Alice Smith", "no", "waiting"],
        ["Bob Jones", "yes", "done"],
    ]
    config = sheet_inference.infer_sheet_config(header, rows, "Sheet1", "unsure")
    assert config.name_col == "Full Name"
    assert "Paid?" in config.decision_cols
