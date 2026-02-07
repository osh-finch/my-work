import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
import urllib.parse

# --- CONFIGURATION ---
# Google Sheet details
SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
SHEET_NAME = 'TO PAY'
NAME_COLUMN = 'Name'
TO_PAY_COLUMN = 'To Pay'
PAID_COLUMN = 'Paid?'

# Contact database details
CONTACTS_FILE = 'contactsdb.csv'
CONTACTS_NAME_COLUMN = 'Name'
CONTACTS_NUMBER_COLUMN = 'Phone'

# Message details
BANK_DETAILS = """
PAYMENT INFO
Account Name: Cambridge University Hockey Club
Account Number: 72372762
Sort Code: 40-16-08
Reference: msubsINITIALSURNAME (eg msubsWSTUBBS)
Due Date: 8th January - late fines of £1/day after this
All information can be found on email - contact ws418 if any issues :)
"""

def get_google_sheet(sheet_url):
    """
    Authenticates with Google and returns the specified worksheet.
    """
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url).worksheet(SHEET_NAME)
        return sheet
    except Exception as e:
        print(f"Error accessing Google Sheet: {e}")
        print("Please ensure 'credentials.json' is set up correctly and the sheet is shared.")
        return None

def get_contacts_data():
    """
    Reads the contacts database from the CSV file.
    """
    try:
        contacts_df = pd.read_csv(CONTACTS_FILE)
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

def send_whatsapp_message(driver, number, message, press_enter=True):
    """
    Sends a WhatsApp message using Selenium.
    """
    try:
        number = str(number)
        if not number.startswith('+'):
            number = '+' + number
        
        url = f"https://web.whatsapp.com/send?phone={number}"
        driver.get(url)
        time.sleep(5) # Give the page time to load

        # Wait for the message box to be ready
        wait = WebDriverWait(driver, 60)
        message_box_xpath = '//*[@id="main"]//div[@contenteditable="true"]'
        print("Waiting for message box...")
        message_box = wait.until(EC.presence_of_element_located((By.XPATH, message_box_xpath)))
        print("Message box found.")
        
        # Type the message and send it
        print("Typing message...")
        lines = message.split('\n')
        for line in lines:
            message_box.send_keys(line)
            message_box.send_keys(Keys.SHIFT, Keys.ENTER)
        
        if press_enter:
            message_box.send_keys(Keys.ENTER)
            print("Message sent.")

        time.sleep(5) # Wait a bit before the next message
        return True
    except TimeoutException:
        print(f"Timed out waiting for page to load for number: {number}. Skipping.")
        return False
    except Exception as e:
        print(f"Could not send message to {number}: {e}")
        return False
