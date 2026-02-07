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
)

from PySide6.QtCore import QThread, Signal, QObject
import chaser_logic

class BatchModeThread(QThread):
    log_signal = Signal(str)

    def __init__(self, sheet_url):
        super().__init__()
        self.sheet_url = sheet_url

    def run(self):
        self.log_signal.emit("Starting batch mode...")
        
        # Get data from Google Sheet
        sheet = chaser_logic.get_google_sheet(self.sheet_url)
        if not sheet:
            self.log_signal.emit("Error: Could not access Google Sheet.")
            return
            
        sheet_values = sheet.get_all_values()
        sheet_df = pd.DataFrame(sheet_values[1:], columns=sheet_values[0])
        
        # Get contacts data
        contacts_df = chaser_logic.get_contacts_data()
        if contacts_df is None:
            self.log_signal.emit("Error: Could not read contacts data.")
            return

        # Merge dataframes
        merged_df = pd.merge(sheet_df, contacts_df, on=chaser_logic.NAME_COLUMN, how='left')

        # Filter for unpaid members
        unpaid_df = merged_df[merged_df[chaser_logic.PAID_COLUMN].str.lower() != 'yes']

        if unpaid_df.empty:
            self.log_signal.emit("No one to chase. Everyone has paid. :)")
            return

        self.log_signal.emit(f"Found {len(unpaid_df)} people to chase.")

        # Initialize the webdriver
        service = chaser_logic.ChromeService()
        driver = chaser_logic.webdriver.Chrome(service=service)
        
        for index, row in unpaid_df.iterrows():
            name = row[chaser_logic.NAME_COLUMN]
            amount = row[chaser_logic.TO_PAY_COLUMN]
            number = row[chaser_logic.CONTACTS_NUMBER_COLUMN]

            if pd.notna(number):
                message = f"""Hi {name}, this is a reminder to please pay your subs of £{amount}.
{chaser_logic.BANK_DETAILS}"""
                
                self.log_signal.emit(f"Sending message to {name} at {number}...")
                if chaser_logic.send_whatsapp_message(driver, number, message):
                    self.log_signal.emit(f"Message sent to {name} successfully.")
                else:
                    self.log_signal.emit(f"Failed to send message to {name}.")
            else:
                self.log_signal.emit(f"Could not find number for {name}. Skipping.")

        self.log_signal.emit("All messages sent. Closing the browser.")
        driver.quit()


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

        self.start_chasing_button.clicked.connect(self.start_batch_mode)
        self.search_input.textChanged.connect(self.search_contacts)
        self.search_results.itemDoubleClicked.connect(self.add_to_mailing_list)
        self.remove_from_mailing_list_button.clicked.connect(self.remove_from_mailing_list)
        self.import_button.clicked.connect(self.import_contacts)
        self.send_message_button.clicked.connect(self.send_individual_message)

        self.load_contacts()

    def setup_batch_mode_ui(self):
        layout = QVBoxLayout()
        self.batch_mode_tab.setLayout(layout)

        # URL input
        url_layout = QHBoxLayout()
        url_label = QLabel("Google Sheet URL:")
        self.sheet_url_input = QLineEdit()
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.sheet_url_input)
        layout.addLayout(url_layout)

        # Start button
        self.start_chasing_button = QPushButton("Start Chasing")
        layout.addWidget(self.start_chasing_button)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        
        # Add a note to the user
        note_label = QLabel("<b>Note:</b> Please make sure you are logged in to WhatsApp Web in your Chrome browser before starting.")
        layout.addWidget(note_label)

    def setup_individual_mode_ui(self):
        main_layout = QHBoxLayout()
        self.individual_mode_tab.setLayout(main_layout)

        # Left side: Search and results
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search Contact:")
        self.search_input = QLineEdit()
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
        main_layout.addLayout(right_layout)

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
        note_label = QLabel("<b>Note:</b> Please make sure you are logged in to WhatsApp Web in your Chrome browser before sending a message.")
        right_layout.addWidget(note_label)

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

    def start_batch_mode(self):
        sheet_url = self.sheet_url_input.text()
        if not sheet_url:
            self.log_output.append("Please enter a Google Sheet URL.")
            return

        self.batch_thread = BatchModeThread(sheet_url)
        self.batch_thread.log_signal.connect(self.update_log)
        self.batch_thread.start()

    def update_log(self, message):
        self.log_output.append(message)

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

        service = chaser_logic.ChromeService()
        driver = chaser_logic.webdriver.Chrome(service=service)

        for i in range(self.mailing_list.count()):
            item = self.mailing_list.item(i)
            selected_name = item.text()
            number = self.contacts_df[self.contacts_df[chaser_logic.CONTACTS_NAME_COLUMN] == selected_name][chaser_logic.CONTACTS_NUMBER_COLUMN].iloc[0]
            
            self.update_log(f"Sending message to {selected_name} at {number}...")
            if chaser_logic.send_whatsapp_message(driver, number, message, press_enter):
                self.update_log(f"Message sent to {selected_name} successfully.")
            else:
                self.update_log(f"Failed to send message to {selected_name}.")

        driver.quit()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
