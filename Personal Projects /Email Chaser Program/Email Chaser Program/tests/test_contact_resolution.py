import pandas as pd

import contact_resolution


def test_normalize_phone_accepts_e164():
    assert contact_resolution.normalize_phone("+447700900123") == "+447700900123"


def test_normalize_phone_rejects_short_number():
    assert contact_resolution.normalize_phone("1234") is None


def test_resolve_uses_sheet_phone_first():
    contacts = pd.DataFrame({"Name": ["Alice"], "Phone": ["+447700900999"]})
    result = contact_resolution.resolve_contact("Alice", "+447700900123", contacts)
    assert result.resolved_phone == "+447700900123"
    assert result.source == "sheet"


def test_resolve_falls_back_to_contacts():
    contacts = pd.DataFrame({"Name": ["Alice"], "Phone": ["+447700900999"]})
    result = contact_resolution.resolve_contact("Alice", None, contacts)
    assert result.resolved_phone == "+447700900999"
    assert result.source == "contacts"


def test_resolve_missing_contact():
    contacts = pd.DataFrame({"Name": ["Alice"], "Phone": ["+447700900999"]})
    result = contact_resolution.resolve_contact("Bob", None, contacts)
    assert result.resolved_phone is None
    assert result.issue == "missing_contact"
