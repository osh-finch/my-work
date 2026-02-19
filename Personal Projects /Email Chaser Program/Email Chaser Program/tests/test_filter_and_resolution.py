import pandas as pd

import cli_chaser


def test_filter_phrase_matching():
    predicate = cli_chaser._build_filter("unsure", ["Decision"])
    row = {"Decision": "I am unsure because selection has not yet occurred"}
    assert predicate(row) is True


def test_resolve_phone_from_contacts():
    contacts = pd.DataFrame({"Name": ["Alice"], "Phone": ["+447700900123"]})
    phone = cli_chaser.resolve_phone("Alice", contacts)
    assert phone == "+447700900123"
