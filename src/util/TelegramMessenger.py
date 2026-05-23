import os
import time
import threading
import queue
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_TOKEN = os.getenv("TELEGRAM_BOT_API_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{API_TOKEN}/sendMessage"

# Internal variables
_msg_queue = queue.Queue()
_worker_thread = None
_lock = threading.Lock()

def _telegram_worker():
    """Background worker that handles the actual sending."""
    while True:
        message = _msg_queue.get()
        if message is None: break
            
        try:
            payload = {"chat_id": CHAT_ID, "text": message}
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                TELEGRAM_API_URL, 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                pass
            
            # Rate limiting to prevent "Connection Closed" errors
            time.sleep(1.1) 
        except Exception as e:
            print(f"Telegram Worker Error: {e}")
        finally:
            _msg_queue.task_done()

def send_telegram_message(message):
    """The only function you import in your main script."""
    global _worker_thread
    
    # Lazy Start: Check if thread is running; if not, start it.
    if _worker_thread is None or not _worker_thread.is_alive():
        with _lock: # Ensure two rapid calls don't start two threads
            if _worker_thread is None or not _worker_thread.is_alive():
                _worker_thread = threading.Thread(target=_telegram_worker, daemon=True)
                _worker_thread.start()
    
    _msg_queue.put(message)