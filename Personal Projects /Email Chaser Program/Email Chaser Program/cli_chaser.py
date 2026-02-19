import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import chaser_logic
import contact_resolution
import history_db
import intent_parser
import llm_decider
import sheet_ingestion
import sheet_inference
import whatsapp_sender
from batch_workflow import format_currency
from cooldown_guard import apply_cooldown_guard
from plan_model import Plan, PlanRecipient


def main():
    parser = argparse.ArgumentParser(description="WhatsApp Chaser Assistant (CLI)")
    parser.add_argument("--cooldown-days", type=int, default=7)
    parser.add_argument("--include-recent", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sheet_url = input("Google Sheet URL or ID: ").strip()
    instruction = input("Instruction: ").strip()

    db_path = _default_db_path()
    history_db.history_db_init(db_path)

    plan = build_plan(sheet_url, instruction, db_path)
    plan = enrich_with_history(plan, db_path)
    plan = apply_cooldown(plan, args.cooldown_days, args.include_recent)

    while True:
        render_plan_preview(plan, args.cooldown_days, args.include_recent)
        choice = approval_prompt()
        if choice == "send":
            if args.dry_run:
                print("Dry run enabled. No messages were sent.")
                return
            send_messages(plan, db_path)
            return
        if choice == "edit_message":
            instruction = input("New instruction for message tone or content: ").strip()
            plan = rebuild_messages(plan, instruction)
            continue
        if choice == "edit_filter":
            instruction = input("New filter instruction: ").strip()
            plan = build_plan(sheet_url, instruction, db_path)
            plan = enrich_with_history(plan, db_path)
            plan = apply_cooldown(plan, args.cooldown_days, args.include_recent)
            continue
        if choice == "export_unresolved":
            export_unresolved(plan)
            continue
        if choice == "override":
            plan = apply_cooldown(plan, args.cooldown_days, include_recent=True)
            continue
        if choice == "cancel":
            print("Cancelled.")
            return


def build_plan(sheet_url: str, instruction: str, db_path: str) -> Plan:
    intent = intent_parser.parse_user_intent(instruction)
    sheet_id = _extract_sheet_id(sheet_url)

    sheet_data = sheet_ingestion.fetch_sheet_rows(sheet_url, None, None)
    header = sheet_data.header
    rows = sheet_data.rows

    cached = history_db.load_cached_sheet_config(db_path, sheet_id)
    if cached:
        tab_name, header_hash, name_col, decision_cols_csv = cached
        if header_hash == _header_hash(header):
            name_col_use = name_col
            decision_cols = decision_cols_csv.split("|") if decision_cols_csv else []
        else:
            name_col_use, decision_cols = None, []
    else:
        name_col_use, decision_cols = None, []

    if not name_col_use or not decision_cols:
        inferred = sheet_inference.infer_sheet_config(
            header, rows, sheet_data.source_sheet, intent.filter_phrase
        )
        name_col_use = inferred.name_col
        decision_cols = inferred.decision_cols
        history_db.cache_sheet_config(
            db_path,
            sheet_id,
            sheet_data.source_sheet,
            _header_hash(header),
            name_col_use,
            "|".join(decision_cols),
        )

    filter_description = intent.filter_phrase or "matches instruction"
    filter_predicate = _build_filter(intent.filter_phrase, decision_cols)

    plan = Plan(
        sheet_id=sheet_id,
        sheet_url=sheet_url,
        tab_name=sheet_data.source_sheet,
        value_range=None,
        name_col=name_col_use,
        decision_cols=decision_cols,
        filter_description=filter_description,
        filter_predicate=filter_predicate,
        objective=intent.message_goal,
        instruction=intent.instruction,
    )

    contacts_df = chaser_logic.get_contacts_data()
    for idx, row in enumerate(rows, start=2):
        row_dict = dict(zip(header, row))
        if not filter_predicate(row_dict):
            continue
        full_name = str(row_dict.get(name_col_use, "")).strip()
        if not full_name:
            plan.unresolved.append(
                PlanRecipient(row_index=idx, full_name="", phone=None, message_text="", unresolved_reason="Blank name")
            )
            continue
        phone = resolve_phone(full_name, contacts_df)
        if not phone:
            plan.unresolved.append(
                PlanRecipient(
                    row_index=idx,
                    full_name=full_name,
                    phone=None,
                    message_text="",
                    unresolved_reason="Name not found in contacts",
                )
            )
            continue
        plan.recipients.append(
            PlanRecipient(
                row_index=idx,
                full_name=full_name,
                phone=phone,
                message_text="",
            )
        )

    plan.recipients = draft_messages(plan, rows, header)
    return plan


def draft_messages(plan: Plan, rows: List[List[str]], header: List[str]) -> List[PlanRecipient]:
    intent = {
        "goal_tone": plan.objective,
        "confirmation_mode": "each",
        "objective": plan.objective,
        "refinement_instructions": plan.instruction,
    }
    column_mapping = {
        "recipient_identifier": plan.name_col,
        "phone_number": "",
        "needs_messaging": ",".join(plan.decision_cols),
        "amount_column": _guess_amount_col(header),
        "personalization_columns": "",
    }

    rows_payload = []
    for idx, row in enumerate(rows, start=2):
        row_dict = dict(zip(header, row))
        if not plan.filter_predicate(row_dict):
            continue
        payload = {
            "row_reference": f"row {idx}",
            plan.name_col: row_dict.get(plan.name_col, ""),
        }
        if column_mapping["amount_column"]:
            amount_val = row_dict.get(column_mapping["amount_column"], "")
            payload[column_mapping["amount_column"]] = amount_val
            payload[f"{column_mapping['amount_column']}_formatted"] = format_currency(amount_val)
        for col in plan.decision_cols:
            payload[col] = row_dict.get(col, "")
        rows_payload.append(payload)

    decisions = llm_decider.decide_messages_with_llm(rows_payload, intent, column_mapping)
    by_name: Dict[str, str] = {d.recipient_identifier: d.message_text for d in decisions}
    for recipient in plan.recipients:
        recipient.message_text = by_name.get(recipient.full_name, recipient.message_text)
    return plan.recipients


def rebuild_messages(plan: Plan, instruction: str) -> Plan:
    plan.instruction = instruction
    sheet_data = sheet_ingestion.fetch_sheet_rows(plan.sheet_url, plan.tab_name, None)
    plan.recipients = draft_messages(plan, sheet_data.rows, sheet_data.header)
    return plan


def resolve_phone(full_name: str, contacts_df) -> Optional[str]:
    if contacts_df is None:
        return None
    name_col = contacts_df.columns[0]
    phone_col = contacts_df.columns[1]
    matches = contacts_df[contacts_df[name_col].str.lower() == full_name.lower()]
    if matches.empty:
        return None
    phone_raw = matches.iloc[0][phone_col]
    return contact_resolution.normalize_phone(phone_raw)


def enrich_with_history(plan: Plan, db_path: str) -> Plan:
    for recipient in plan.recipients:
        last = history_db.history_last_message_for_phone(db_path, recipient.phone)
        if not last:
            recipient.last_messaged_relative = "never"
            continue
        sent_at, message_text = last
        recipient.last_messaged_at = sent_at
        recipient.last_messaged_relative = _relative_time(sent_at)
        recipient.last_message_snippet = _truncate(message_text, 80)
    return plan


def apply_cooldown(plan: Plan, cooldown_days: int, include_recent: bool) -> Plan:
    included, excluded = apply_cooldown_guard(plan.recipients, cooldown_days, include_recent)
    plan.recipients = included
    plan.excluded_recent = excluded
    return plan


def render_plan_preview(plan: Plan, cooldown_days: int, include_recent: bool) -> None:
    print("\nPlan preview")
    print(f"Matched rows: {len(plan.recipients) + len(plan.unresolved) + len(plan.excluded_recent)}")
    print(f"Resolved phones: {len(plan.recipients)}")
    print(f"Unresolved: {len(plan.unresolved)}")
    print(f"Excluded due to cooldown: {len(plan.excluded_recent)} (cooldown {cooldown_days} days)")

    print("\nRecipients (first 10):")
    for recipient in plan.recipients[:10]:
        phone_masked = _mask_phone(recipient.phone)
        last_msg = recipient.last_messaged_relative or "never"
        snippet = f" | Last message: {recipient.last_message_snippet}" if recipient.last_message_snippet else ""
        print(
            f"- {recipient.full_name} ({phone_masked}) | Last messaged: {last_msg}{snippet}"
        )
        print(f"  Message: {recipient.message_text}")

    if plan.unresolved:
        print("\nUnresolved:")
        for item in plan.unresolved[:10]:
            name = item.full_name or "(blank name)"
            print(f"- {name}: {item.unresolved_reason}")

    if plan.excluded_recent and not include_recent:
        print("\nRepeat message risk:")
        for item in plan.excluded_recent[:10]:
            print(f"- {item.full_name}: last messaged {item.last_messaged_relative}")


def approval_prompt() -> str:
    print("\nApproval")
    print("1) Send now")
    print("2) Edit message")
    print("3) Edit filter")
    print("4) Export unresolved list")
    print("5) Override repeat guard")
    print("6) Cancel")
    choice = input("Choose: ").strip()
    return {
        "1": "send",
        "2": "edit_message",
        "3": "edit_filter",
        "4": "export_unresolved",
        "5": "override",
        "6": "cancel",
    }.get(choice, "cancel")


def send_messages(plan: Plan, db_path: str) -> None:
    driver = whatsapp_sender.build_whatsapp_driver()
    try:
        for recipient in plan.recipients:
            ok = whatsapp_sender.send_message_via_whatsapp_web(
                driver,
                recipient.phone,
                recipient.message_text,
                press_enter=True,
            )
            if ok:
                history_db.history_write_sent(
                    db_path,
                    recipient.phone,
                    recipient.message_text,
                    plan.sheet_id,
                    f"row {recipient.row_index}",
                    plan.instruction,
                )
            else:
                history_db.history_write_failed(
                    db_path,
                    recipient.phone,
                    recipient.message_text,
                    plan.sheet_id,
                    f"row {recipient.row_index}",
                    plan.instruction,
                    "send_failed",
                )
    finally:
        if driver:
            driver.quit()


def export_unresolved(plan: Plan) -> None:
    path = "unresolved.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("row_index,full_name,reason\n")
        for item in plan.unresolved:
            handle.write(f"{item.row_index},{item.full_name},{item.unresolved_reason}\n")
    print(f"Exported unresolved list to {path}")


def _default_db_path() -> str:
    return os.path.join(".app_state", "history.db")


def _extract_sheet_id(sheet_url: str) -> str:
    if "/d/" in sheet_url:
        return sheet_url.split("/d/")[1].split("/")[0]
    return sheet_url


def _header_hash(header: List[str]) -> str:
    return "|".join(header)


def _build_filter(phrase: Optional[str], decision_cols: List[str]):
    def predicate(row: dict) -> bool:
        if not decision_cols:
            return False
        values = " ".join(str(row.get(col, "")) for col in decision_cols).lower()
        if phrase:
            return phrase.lower() in values
        return bool(values.strip())

    return predicate


def _guess_amount_col(header: List[str]) -> str:
    lower = [h.lower() for h in header]
    for cand in ["to pay", "amount", "amount owed", "balance"]:
        if cand in lower:
            return header[lower.index(cand)]
    return ""


def _relative_time(sent_at_iso: str) -> str:
    try:
        sent_at = datetime.fromisoformat(sent_at_iso)
    except ValueError:
        return "unknown"
    now = datetime.now(timezone.utc)
    delta = now - sent_at
    days = delta.days
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    return f"{days} days ago"


def _mask_phone(phone: Optional[str]) -> str:
    if not phone:
        return "unknown"
    if len(phone) <= 4:
        return phone
    return f"{phone[:3]}****{phone[-2:]}"


def _truncate(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


if __name__ == "__main__":
    main()
