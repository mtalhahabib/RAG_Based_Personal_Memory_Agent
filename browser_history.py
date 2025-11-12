import sqlite3
import os
import shutil
import time
from dotenv import load_dotenv

load_dotenv()
EVENT_DB = os.environ.get("EVENT_DB", "events.db")
CHROME_HISTORY_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\History")
MAX_ENTRIES = 200

def fetch_recent_history(days=3):
    if not os.path.exists(CHROME_HISTORY_PATH):
        return []

    temp_copy = "browser_history_temp.db"
    if os.path.exists(temp_copy):
        os.remove(temp_copy)
    shutil.copy2(CHROME_HISTORY_PATH, temp_copy)

    conn = sqlite3.connect(temp_copy)
    c = conn.cursor()
    c.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT ?", (MAX_ENTRIES,))
    rows = c.fetchall()
    conn.close()

    def chrome_to_unix(chrome_ts):
        return int((chrome_ts - 11644473600000000) / 1_000_000)

    entries = []
    for url, title, visit_time in rows:
        ts = chrome_to_unix(visit_time)
        if time.time() - ts <= days * 86400:
            entries.append((url, title, ts))

    conn2 = sqlite3.connect(EVENT_DB)
    c2 = conn2.cursor()
    c2.execute("""
        CREATE TABLE IF NOT EXISTS browser_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT,
            visit_time REAL
        )
    """)
    for url, title, ts in entries:
        c2.execute("INSERT INTO browser_history (url, title, visit_time) VALUES (?, ?, ?)", (url, title, ts))
    conn2.commit()
    conn2.close()

    return [f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] {title} — {url}" for url, title, ts in entries]

def get_recent_browser_history(limit=10, days=3):
    entries = fetch_recent_history(days=days)
    return entries[:limit]
