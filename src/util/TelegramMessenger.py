from telegram import Bot
from dotenv import load_dotenv
import requests

import os


# check if there are setting in a .env file

load_dotenv()
# Replace with your bot's API token
API_TOKEN = os.getenv("TELEGRAM_BOT_API_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MESSAGE = os.getenv("TELEGRAM_DEFAULT_MESSAGE", "No default set")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{API_TOKEN}/sendMessage"

def send_telegram_message(message=MESSAGE):
    # bot = Bot(token=API_TOKEN)
    # bot.send_message(chat_id=CHAT_ID, text=MESSAGE)
    # print("Message sent!")
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(TELEGRAM_API_URL, data=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to send message: {response.text}")
    return response.json()

if __name__ == "__main__":
    # send_telegram_message("Test message from Telegram bot!")
    print(MESSAGE)
    send_telegram_message()
    send_telegram_message("Another test message from Telegram bot!")
