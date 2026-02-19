import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QListWidget,
    QLabel,
    QAbstractItemView,
    QFileDialog,
    QCheckBox,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
)

from PySide6.QtCore import QThread, Signal, QObject
import batch_workflow
import chaser_logic
import llm_batch_ui
import llm_decider
import sheet_ingestion
import whatsapp_sender
from contact_resolution import normalize_phone

class LlmDecisionThread(QThread):
    decisions_ready = Signal(list, dict)
    error_signal = Signal(str)

    def __init__(self, state: llm_batch_ui.BatchFlowState, sheet_data=None):
        super().__init__()
        self.state = state
        self.sheet_data = sheet_data

    def run(self):
        try:
            sheet_data = self.sheet_data or sheet_ingestion.fetch_sheet_rows(
                self.state.sheet_url_or_id,
                self.state.worksheet_name,
                value_range=self.state.value_range,
            )
            rows_payload, row_lookup = batch_workflow.build_rows_payload(
                sheet_data.header,
                sheet_data.rows,
                sheet_data.row_numbers,
                self.state.recipient_column,
                self.state.phone_column,
                self.state.needs_columns or [],
                self.state.amount_column,
                self.state.personalization_columns or [],
            )

            intent = {
                "goal_tone": self.state.goal_tone,
                "confirmation_mode": self.state.confirmation_mode,
                "objective": self.state.objective,
                "refinement_instructions": self.state.refinement_notes,
            }
            column_mapping = {
                "recipient_identifier": self.state.recipient_column,
                "phone_number": self.state.phone_column or "",
                "needs_messaging": ",".join(self.state.needs_columns or []),
                "amount_column": self.state.amount_column or "",
                "personalization_columns": ",".join(self.state.personalization_columns or []),
            }

            decisions = llm_decider.decide_messages_with_llm(
                rows_payload,
                intent,
                column_mapping,
            )

            self.decisions_ready.emit(decisions, row_lookup)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class LlmInferenceThread(QThread):
    inference_ready = Signal(object, object)
    error_signal = Signal(str)

    def __init__(self, state: llm_batch_ui.BatchFlowState):
        super().__init__()
        self.state = state

    def run(self):
        try:
            sheet_data = sheet_ingestion.fetch_sheet_rows(
                self.state.sheet_url_or_id,
                self.state.worksheet_name,
                value_range=self.state.value_range,
            )
            summary = batch_workflow.summarize_sheet(sheet_data.header, sheet_data.rows)
            try:
                inference = llm_decider.infer_sheet_context(summary)
            except Exception as exc:
                self.error_signal.emit(f"LLM inference failed: {exc}")
                inference = None
            self.inference_ready.emit(sheet_data, inference)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class LlmSendThread(QThread):
    log_signal = Signal(str)
    done_signal = Signal()

    def __init__(self, decisions: list):
        super().__init__()
        self.decisions = decisions

    def run(self):
        driver = whatsapp_sender.build_whatsapp_driver()
        try:
            for decision in self.decisions:
                self.log_signal.emit(
                    f"Sending to {decision.recipient_identifier} at {decision.resolved_phone}..."
                )
                ok = whatsapp_sender.send_message_via_whatsapp_web(
                    driver,
                    decision.resolved_phone,
                    decision.message_text,
                    press_enter=True,
                )
                if ok:
                    self.log_signal.emit(f"Message sent to {decision.recipient_identifier}.")
                else:
                    self.log_signal.emit(f"Failed to send message to {decision.recipient_identifier}.")
        finally:
            if driver:
                driver.quit()
        self.done_signal.emit()


import pandas as pd
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhatsApp Chaser")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.batch_mode_tab = QWidget()
        self.individual_mode_tab = QWidget()

        self.tabs.addTab(self.batch_mode_tab, "Batch Mode")
        self.tabs.addTab(self.individual_mode_tab, "Individual Mode")

        self.setup_batch_mode_ui()
        self.setup_individual_mode_ui()
        self.apply_component_styles()

        self.search_input.textChanged.connect(self.search_contacts)
        self.search_results.itemDoubleClicked.connect(self.add_to_mailing_list)
        self.remove_from_mailing_list_button.clicked.connect(self.remove_from_mailing_list)
        self.import_button.clicked.connect(self.import_contacts)
        self.send_message_button.clicked.connect(self.send_individual_message)
        self.chat_send_button.clicked.connect(self.handle_chat_send)
        self.chat_input.returnPressed.connect(self.handle_chat_send)
        self.preview_approve_button.clicked.connect(self.preview_approve_selected)
        self.preview_skip_button.clicked.connect(self.preview_skip_selected)
        self.preview_edit_button.clicked.connect(self.preview_edit_selected)
        self.preview_approve_all_button.clicked.connect(self.preview_approve_all)
        self.preview_send_button.clicked.connect(self.preview_send_approved)
        self.preview_list.currentRowChanged.connect(self.preview_show_selected)

        self.load_contacts()
        self.init_batch_chat()

    def apply_component_styles(self):
        self.send_message_button.setProperty("variant", "primary")
        self.chat_send_button.setProperty("variant", "primary")
        self.preview_send_button.setProperty("variant", "primary")
        self.import_button.setProperty("variant", "secondary")
        self.remove_from_mailing_list_button.setProperty("variant", "secondary")
        self.preview_approve_button.setProperty("variant", "secondary")
        self.preview_skip_button.setProperty("variant", "secondary")
        self.preview_edit_button.setProperty("variant", "secondary")
        self.preview_approve_all_button.setProperty("variant", "secondary")

        self.batch_mode_note_label.setProperty("variant", "note")
        self.individual_mode_note_label.setProperty("variant", "note")

    def setup_batch_mode_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        self.batch_mode_tab.setLayout(layout)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        layout.addLayout(left_layout, 2)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        layout.addLayout(right_layout, 1)

        title_label = QLabel("Batch Mode")
        title_label.setProperty("variant", "title")
        left_layout.addWidget(title_label)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        left_layout.addWidget(self.chat_log)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_send_button = QPushButton("Send")
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.chat_send_button)
        left_layout.addLayout(chat_input_layout)

        self.batch_mode_note_label = QLabel(
            "<b>Note:</b> Please make sure you are logged in to WhatsApp Web in your Chrome browser before sending."
        )
        left_layout.addWidget(self.batch_mode_note_label)

        preview_title = QLabel("Preview")
        preview_title.setProperty("variant", "section")
        right_layout.addWidget(preview_title)

        self.preview_list = QListWidget()
        right_layout.addWidget(self.preview_list)

        preview_detail_label = QLabel("Selected Message")
        preview_detail_label.setProperty("variant", "section")
        right_layout.addWidget(preview_detail_label)

        self.preview_detail = QTextEdit()
        self.preview_detail.setReadOnly(True)
        right_layout.addWidget(self.preview_detail)

        preview_buttons = QHBoxLayout()
        self.preview_approve_button = QPushButton("Approve")
        self.preview_skip_button = QPushButton("Skip")
        self.preview_edit_button = QPushButton("Edit")
        preview_buttons.addWidget(self.preview_approve_button)
        preview_buttons.addWidget(self.preview_skip_button)
        preview_buttons.addWidget(self.preview_edit_button)
        right_layout.addLayout(preview_buttons)

        self.preview_approve_all_button = QPushButton("Approve All")
        self.preview_send_button = QPushButton("Send Approved")
        right_layout.addWidget(self.preview_approve_all_button)
        right_layout.addWidget(self.preview_send_button)

    def setup_individual_mode_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        self.individual_mode_tab.setLayout(main_layout)

        # Left side: Search and results
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        main_layout.addLayout(left_layout)

        left_title = QLabel("Contacts")
        left_title.setProperty("variant", "section")
        left_layout.addWidget(left_title)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        search_label = QLabel("Search Contact:")
        self.search_input = QLineEdit()
        self.search_input.setMinimumWidth(self.search_input.fontMetrics().averageCharWidth() * 25 + 24)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        left_layout.addLayout(search_layout)

        # Import button
        self.import_button = QPushButton("Import Contacts")
        left_layout.addWidget(self.import_button)

        # Search results
        self.search_results = QListWidget()
        left_layout.addWidget(self.search_results)

        # Right side: Mailing list and message
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        main_layout.addLayout(right_layout)

        right_title = QLabel("Message")
        right_title.setProperty("variant", "section")
        right_layout.addWidget(right_title)

        # Mailing list
        mailing_list_label = QLabel("Mailing List:")
        self.mailing_list = QListWidget()
        right_layout.addWidget(mailing_list_label)
        right_layout.addWidget(self.mailing_list)

        # Remove button
        self.remove_from_mailing_list_button = QPushButton("Remove from Mailing List")
        right_layout.addWidget(self.remove_from_mailing_list_button)

        # Message input
        message_label = QLabel("Message:")
        self.message_input = QTextEdit()
        right_layout.addWidget(message_label)
        right_layout.addWidget(self.message_input)

        # Press enter checkbox
        self.press_enter_checkbox = QCheckBox("Press Enter to send")
        self.press_enter_checkbox.setChecked(True)
        right_layout.addWidget(self.press_enter_checkbox)

        # Send button
        self.send_message_button = QPushButton("Send Message")
        right_layout.addWidget(self.send_message_button)

        # Add a note to the user
        self.individual_mode_note_label = QLabel("<b>Note:</b> Please make sure you are logged in to WhatsApp Web in your Chrome browser before sending a message.")
        right_layout.addWidget(self.individual_mode_note_label)

    def import_contacts(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                new_contacts_df = pd.read_csv(file_path, header=None)
                new_contacts_df.columns = [chaser_logic.CONTACTS_NAME_COLUMN, chaser_logic.CONTACTS_NUMBER_COLUMN]
                
                if self.contacts_df is None:
                    self.contacts_df = new_contacts_df
                else:
                    self.contacts_df = pd.concat([self.contacts_df, new_contacts_df], ignore_index=True)
                
                self.search_results.clear()
                self.search_results.addItems(self.contacts_df[chaser_logic.CONTACTS_NAME_COLUMN].tolist())
                self.update_log(f"Successfully imported contacts from {file_path}")
            except Exception as e:
                self.update_log(f"Error importing contacts: {e}")

    def update_log(self, message):
        if hasattr(self, "chat_log"):
            self.chat_log.append(f"<b>System:</b> {message}")

    def init_batch_chat(self):
        self.batch_chat_flow = llm_batch_ui.LlmBatchChatFlow()
        self.batch_chat_state = self.batch_chat_flow.state
        self.sheet_data_cache = None
        self.preview_decisions = []
        self.preview_status = {}
        self.append_chat_message("assistant", self.get_time_greeting())
        self.append_chat_message("assistant", self.batch_chat_flow.start_prompt())

    def get_time_greeting(self):
        from datetime import datetime

        hour = datetime.now().hour
        if hour < 12:
            return "Good morning! I can help draft your chase messages."
        if hour < 18:
            return "Good afternoon! I can help draft your chase messages."
        return "Good evening! I can help draft your chase messages."

    def append_chat_message(self, role, text):
        prefix = "You" if role == "user" else "Assistant"
        self.chat_log.append(f"<b>{prefix}:</b> {text}")

    def handle_chat_send(self):
        text = self.chat_input.text().strip()
        if not text and not self.batch_chat_flow.allows_empty_input():
            return
        display_text = text if text else "(blank)"
        self.append_chat_message("user", display_text)
        self.chat_input.clear()
        if text.lower() == "adjust":
            self.batch_chat_flow.start_manual_override()
            self.append_chat_message(
                "assistant",
                "Sure — tell me in one message which columns to use (recipient, needs-chasing, and optional phone/amount).",
            )
            return
        if self.batch_chat_flow._awaiting_freeform_config:
            prompt = self.batch_chat_flow.handle_freeform_config(text)
            if prompt:
                self.append_chat_message("assistant", prompt)
            else:
                self.append_chat_message("assistant", "Thanks — got it.")
            return
        next_prompt = self.batch_chat_flow.handle_input(text)
        if next_prompt == "__REFINE_DRAFTS__":
            self.append_chat_message("assistant", "Got it. Regenerating drafts with your refinement...")
            self.batch_chat_flow.disable_refinement_mode()
            self.run_llm_decision()
            return
        if next_prompt == "__FETCH_SUGGESTIONS__":
            self.fetch_suggestions_and_prompt()
            return
        if next_prompt is None:
            self.append_chat_message("assistant", "Thanks. Fetching sheet data and drafting messages...")
            self.run_llm_decision()
        else:
            self.append_chat_message("assistant", next_prompt)

    def run_llm_decision(self):
        self.decision_thread = LlmDecisionThread(self.batch_chat_state, sheet_data=self.sheet_data_cache)
        self.decision_thread.decisions_ready.connect(self.handle_decisions_ready)
        self.decision_thread.error_signal.connect(self.handle_decision_error)
        self.decision_thread.start()

    def fetch_suggestions_and_prompt(self):
        self.append_chat_message("assistant", "Let me read the sheet and suggest columns...")
        self.inference_thread = LlmInferenceThread(self.batch_chat_state)
        self.inference_thread.inference_ready.connect(self.handle_inference_ready)
        self.inference_thread.error_signal.connect(self.handle_inference_error)
        self.inference_thread.start()

    def handle_inference_ready(self, sheet_data, inference):
        self.sheet_data_cache = sheet_data
        summary = batch_workflow.summarize_sheet(sheet_data.header, sheet_data.rows)
        columns_preview = ", ".join(summary["columns"][:8])
        self.append_chat_message(
            "assistant",
            f"I found {summary['row_count']} rows. Columns include: {columns_preview}.",
        )
        if inference and inference.get("objective"):
            confidence = float(inference.get("objective_confidence", 0))
            confidence_note = " (low confidence)" if confidence < 0.5 else ""
            self.append_chat_message(
                "assistant",
                f"I think this is about: {inference.get('objective')}{confidence_note}.",
            )
            if confidence < 0.5:
                self.append_chat_message(
                    "assistant",
                    "I'm not fully confident — please review the suggested columns and adjust if needed.",
                )

        prompt = self.batch_chat_flow.apply_inference(sheet_data.header, inference)
        self.append_chat_message("assistant", prompt)

    def handle_inference_error(self, message):
        self.append_chat_message("assistant", f"Could not infer from sheet: {message}")
        self.batch_chat_flow.restart_worksheet_prompt()
        self.append_chat_message(
            "assistant",
            "Please re-enter the worksheet/tab name (case-insensitive), or press Enter to use the first tab.",
        )

    def handle_decisions_ready(self, decisions, row_lookup):
        contacts_df = chaser_logic.get_contacts_data()
        enriched = batch_workflow.apply_contact_resolution(
            decisions,
            row_lookup=row_lookup,
            phone_column=self.batch_chat_state.phone_column,
            contacts_df=contacts_df,
            log_fn=lambda msg: self.append_chat_message("assistant", msg),
        )
        actionable = [d for d in enriched if d.should_message and d.resolved_phone]
        if not actionable:
            self.append_chat_message("assistant", "No actionable messages after resolution.")
            return
        self.preview_decisions = actionable
        if self.batch_chat_state.confirmation_mode == "auto":
            self.preview_status = {id(d): "approved" for d in actionable}
        else:
            self.preview_status = {id(d): "pending" for d in actionable}
        self.render_preview_list()
        self.append_chat_message("assistant", f"Drafted {len(actionable)} messages. Review in the preview list.")
        self.append_chat_message(
            "assistant",
            "Select a draft to see full details. Approve/Skip/Edit, then click Send Approved to open WhatsApp.",
        )
        self.append_chat_message(
            "assistant",
            "If you want to tweak tone or content, type your instructions here and I’ll regenerate.",
        )
        self.batch_chat_flow.enable_refinement_mode()

    def handle_decision_error(self, message):
        self.append_chat_message("assistant", f"Batch mode failed: {message}")

    def render_preview_list(self):
        self.preview_list.clear()
        for decision in self.preview_decisions:
            status = self.preview_status.get(id(decision), "pending").upper()
            snippet = decision.message_text.replace("\n", " ")
            if len(snippet) > 48:
                snippet = snippet[:48] + "…"
            label = (
                f"{status} | {decision.recipient_identifier} | {decision.row_reference} | "
                f"{decision.resolved_phone} | {snippet}"
            )
            self.preview_list.addItem(label)
        if self.preview_decisions:
            self.preview_list.setCurrentRow(0)

    def get_selected_decision(self):
        idx = self.preview_list.currentRow()
        if idx < 0:
            return None
        return self.preview_decisions[idx]

    def preview_show_selected(self):
        decision = self.get_selected_decision()
        if not decision:
            self.preview_detail.clear()
            return
        detail = (
            f"Recipient: {decision.recipient_identifier}\n"
            f"Row: {decision.row_reference}\n"
            f"Phone: {decision.resolved_phone}\n"
            f"Reason: {decision.reason}\n"
            f"Confidence: {decision.confidence:.2f}\n\n"
            f"{decision.message_text}"
        )
        self.preview_detail.setPlainText(detail)

    def preview_approve_selected(self):
        decision = self.get_selected_decision()
        if not decision:
            return
        self.preview_status[id(decision)] = "approved"
        self.render_preview_list()

    def preview_skip_selected(self):
        decision = self.get_selected_decision()
        if not decision:
            return
        self.preview_status[id(decision)] = "skipped"
        self.render_preview_list()

    def preview_edit_selected(self):
        decision = self.get_selected_decision()
        if not decision:
            return
        edited = self.open_edit_dialog(decision.message_text)
        if edited is None:
            return
        decision.message_text = edited
        self.preview_status[id(decision)] = "approved"
        self.render_preview_list()

    def open_edit_dialog(self, current_text):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Message")
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        editor = QTextEdit()
        editor.setPlainText(current_text)
        layout.addWidget(editor)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            return editor.toPlainText().strip()
        return None

    def preview_approve_all(self):
        for decision in self.preview_decisions:
            self.preview_status[id(decision)] = "approved"
        self.render_preview_list()

    def preview_send_approved(self):
        approved = [
            d for d in self.preview_decisions if self.preview_status.get(id(d)) == "approved"
        ]
        if not approved:
            self.append_chat_message("assistant", "No approved messages to send.")
            return
        self.append_chat_message("assistant", f"Sending {len(approved)} approved messages...")
        self.send_thread = LlmSendThread(approved)
        self.send_thread.log_signal.connect(lambda msg: self.append_chat_message("assistant", msg))
        self.send_thread.done_signal.connect(
            lambda: self.append_chat_message("assistant", "Send complete.")
        )
        self.send_thread.start()

    def load_contacts(self):
        self.contacts_df = chaser_logic.get_contacts_data()
        if self.contacts_df is not None:
            self.search_results.addItems(self.contacts_df[chaser_logic.CONTACTS_NAME_COLUMN].tolist())

    def search_contacts(self, text):
        if self.contacts_df is not None:
            self.search_results.clear()
            filtered_contacts = self.contacts_df[self.contacts_df[chaser_logic.CONTACTS_NAME_COLUMN].str.contains(text, case=False)]
            self.search_results.addItems(filtered_contacts[chaser_logic.CONTACTS_NAME_COLUMN].tolist())

    def add_to_mailing_list(self, item):
        self.mailing_list.addItem(item.text())

    def remove_from_mailing_list(self):
        selected_items = self.mailing_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.mailing_list.takeItem(self.mailing_list.row(item))

    def send_individual_message(self):
        if self.mailing_list.count() == 0:
            self.update_log("Please add at least one contact to the mailing list.")
            return

        message = self.message_input.toPlainText()
        if not message:
            self.update_log("Please enter a message to send.")
            return

        press_enter = self.press_enter_checkbox.isChecked()

        recipients = []
        skipped = []
        for i in range(self.mailing_list.count()):
            item = self.mailing_list.item(i)
            selected_name = item.text()
            number = self.contacts_df[self.contacts_df[chaser_logic.CONTACTS_NAME_COLUMN] == selected_name][chaser_logic.CONTACTS_NUMBER_COLUMN].iloc[0]
            normalized_number = normalize_phone(number)
            if not normalized_number:
                self.update_log(f"Invalid phone number for {selected_name}: {number}")
                print(f"[individual] Invalid phone number for {selected_name}: {number}")
                skipped.append((selected_name, number))
                continue
            recipients.append((selected_name, normalized_number))

        if skipped:
            print("[individual] Skipped recipients due to invalid numbers:")
            for name, raw_number in skipped:
                print(f"[individual]   - {name}: {raw_number}")

        if not recipients:
            self.update_log("No valid phone numbers to send.")
            print("[individual] No valid phone numbers to send.")
            return

        self.update_log("Starting Chrome driver...")
        print("[individual] Starting Chrome driver...")
        driver = whatsapp_sender.build_whatsapp_driver()
        if driver is None:
            self.update_log("Failed to start Chrome driver. Check console for errors.")
            print("[individual] Failed to start Chrome driver.")
            return
        self.update_log("Chrome driver started.")
        print("[individual] Chrome driver started.")

        for selected_name, normalized_number in recipients:
            self.update_log(f"Sending message to {selected_name} at {normalized_number}...")
            print(f"[individual] Sending message to {selected_name} at {normalized_number}...")
            if whatsapp_sender.send_message_via_whatsapp_web(
                driver,
                normalized_number,
                message,
                press_enter,
            ):
                self.update_log(f"Message sent to {selected_name} successfully.")
                print(f"[individual] Message sent to {selected_name}.")
            else:
                self.update_log(f"Failed to send message to {selected_name}.")
                print(f"[individual] Failed to send message to {selected_name}.")

        driver.quit()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        /* Base tokens */
        QWidget {
            background-color: #f7f8fa;
            color: #111827;
            font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial";
            font-size: 14px;
        }

        QLabel {
            font-size: 13px;
        }

        QLabel[variant="title"] {
            font-size: 18px;
            font-weight: 700;
            color: #111827;
        }

        QLabel[variant="section"] {
            font-size: 15px;
            font-weight: 600;
            color: #111827;
        }

        QLineEdit, QTextEdit, QListWidget {
            background-color: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 8px 10px;
            selection-background-color: #2563eb;
            selection-color: #ffffff;
        }

        QLineEdit:focus, QTextEdit:focus, QListWidget:focus {
            border: 1px solid #2563eb;
            outline: none;
        }

        QPushButton {
            background-color: #ffffff;
            color: #111827;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: #f3f4f6;
        }

        QPushButton:pressed {
            background-color: #e5e7eb;
        }

        QPushButton[variant="primary"] {
            background-color: #2563eb;
            color: #ffffff;
            border: 1px solid #1d4ed8;
        }

        QPushButton[variant="primary"]:hover {
            background-color: #1d4ed8;
        }

        QPushButton[variant="primary"]:pressed {
            background-color: #1e40af;
        }

        QPushButton[variant="secondary"] {
            background-color: #ffffff;
            color: #111827;
            border: 1px solid #d1d5db;
        }

        QPushButton[variant="secondary"]:hover {
            background-color: #f3f4f6;
        }

        QPushButton[variant="secondary"]:pressed {
            background-color: #e5e7eb;
        }

        QLabel[variant="note"] {
            color: #6b7280;
            font-size: 12px;
        }

        QCheckBox {
            spacing: 8px;
            font-size: 13px;
        }

        QListWidget::item {
            padding: 6px 8px;
        }

        QListWidget::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }

        QTabWidget::pane {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #ffffff;
            padding: 8px;
        }

        QTabBar::tab {
            background: #e5e7eb;
            color: #374151;
            padding: 8px 14px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 6px;
        }

        QTabBar::tab:selected {
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-bottom: none;
        }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
