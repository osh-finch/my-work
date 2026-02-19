import pandas as pd
# --- CONFIGURATION ---
# Contact database details
CONTACTS_FILE = 'contactsdb.xlsx'
CONTACTS_NAME_COLUMN = 'Name'
CONTACTS_NUMBER_COLUMN = 'Phone'

def get_contacts_data():
    """
    Reads the contacts database from the CSV file.
    """
    try:
        # Read as strings to avoid numeric rounding/precision loss.
        contacts_df = pd.read_excel(CONTACTS_FILE, dtype=str)
        # Use the correct columns for Name and Phone Number
        contacts_df = contacts_df.iloc[:, [0, 3]]
        contacts_df.columns = [CONTACTS_NAME_COLUMN, CONTACTS_NUMBER_COLUMN]
        return contacts_df
    except FileNotFoundError:
        print(f"Error: '{CONTACTS_FILE}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
