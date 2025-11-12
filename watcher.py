import time
import os
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv()
WATCH_PATHS = [p.strip() for p in os.environ.get("WATCH_PATHS", "").split(",") if p.strip()]
EXCLUDE_PATTERNS = [e.strip().lower() for e in os.environ.get("EXCLUDE_PATTERNS", "").split(",")]
EVENT_DB = os.environ.get("EVENT_DB", "events.db")
DEBOUNCE_SEC = float(os.environ.get("DEBOUNCE_SEC", "0.5"))

def init_db():
    conn = sqlite3.connect(EVENT_DB, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            path TEXT,
            timestamp REAL,
            processed INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn

def insert_event(conn, event_type, path, timestamp):
    c = conn.cursor()
    c.execute("INSERT INTO events (event_type, path, timestamp, processed) VALUES (?, ?, ?, 0)",
              (event_type, path, timestamp))
    conn.commit()

def get_recent_file_events(limit=20):
    conn = sqlite3.connect(EVENT_DB)
    c = conn.cursor()
    rows = c.execute("SELECT event_type, path, timestamp FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] {etype.upper()}: {path}" for etype, path, ts in rows]

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, conn):
        self.conn = conn

    def on_any_event(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if any(ex in path.lower() for ex in EXCLUDE_PATTERNS):
            return
        if path.endswith(".db") or path.endswith(".db-journal"):
            return
        insert_event(self.conn, event.event_type, path, time.time())

if __name__ == "__main__":
    conn = init_db()
    observers = []
    for watch_path in WATCH_PATHS:
        if not os.path.exists(watch_path):
            continue
        observer = Observer()
        observer.schedule(FileChangeHandler(conn), watch_path, recursive=True)
        observer.start()
        observers.append(observer)
        print(f"👀 Watching: {watch_path}")

    try:
        while True:
            time.sleep(DEBOUNCE_SEC)
    except KeyboardInterrupt:
        for o in observers:
            o.stop()
        for o in observers:
            o.join()
