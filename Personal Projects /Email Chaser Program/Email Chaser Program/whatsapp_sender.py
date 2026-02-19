from typing import Optional

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException


def send_message_via_whatsapp_web(
    driver,
    phone_number: str,
    message: str,
    press_enter: bool = True,
) -> bool:
    """
    Thin wrapper around Selenium-based WhatsApp Web sending.
    Implementation moved from chaser_logic in Phase 2.
    """
    try:
        number = str(phone_number).strip()
        if number.startswith("+"):
            number = number[1:]
        number = "".join(ch for ch in number if ch.isdigit())
        if not number:
            return False

        url = f"https://web.whatsapp.com/send/?phone={number}&type=phone_number&app_absent=0"
        driver.get(url)
        time.sleep(5)

        wait = WebDriverWait(driver, 60)
        message_box_xpath = '//*[@id="main"]//div[@contenteditable="true"]'
        message_box = wait.until(EC.presence_of_element_located((By.XPATH, message_box_xpath)))

        lines = message.split("\n")
        for line in lines:
            message_box.send_keys(line)
            message_box.send_keys(Keys.SHIFT, Keys.ENTER)

        if press_enter:
            message_box.send_keys(Keys.ENTER)

        time.sleep(5)
        return True
    except TimeoutException:
        return False
    except Exception:
        return False


def build_whatsapp_driver() -> Optional[object]:
    """
    Construct and return a Selenium driver. Implementation added in Phase 2.
    """
    service = ChromeService(log_output="chromedriver.log")
    options = Options()
    # Keep Chrome open if the script exits so we can see failures.
    options.add_experimental_option("detach", True)
    try:
        print("[whatsapp_sender] Starting Chrome driver...")
        return webdriver.Chrome(service=service, options=options)
    except WebDriverException as exc:
        print(f"[whatsapp_sender] WebDriverException: {exc}")
        return None
    except Exception as exc:
        print(f"[whatsapp_sender] Failed to start Chrome driver: {exc}")
        return None
