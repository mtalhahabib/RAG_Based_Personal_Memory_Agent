# indexer.py
"""
Indexer for RAG: reads events.db, extracts text, creates chunk embeddings, upserts into VectorStore.
Run:
    python indexer.py --once
    python indexer.py        (continuous loop)
"""
import os
import time
import argparse
import sqlite3
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from llm_client import embed, get_client
from vectorstore import VectorStore
from utils import read_text_file, chunk_text, sha256_of_text, ensure_dir
from helpers.extract_pdf import extract_pdf_text
from helpers.extract_docx import extract_docx_text

EVENT_DB = os.environ.get("EVENT_DB", "events.db")
POLL_INTERVAL = float(os.environ.get("INDEXER_POLL", 2.0))
CHUNK_SIZE = int(os.environ.get("INDEXER_CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("INDEXER_CHUNK_OVERLAP", 200))
MIN_TOKEN_CHUNKS = int(os.environ.get("MIN_TOKEN_CHUNKS", 1))
EMBED_DIM = int(os.environ.get("EMBED_DIM", 3072))

SUPPORTED = {
    ".txt": read_text_file,
    ".md": read_text_file,
    ".py": read_text_file,
    ".js": read_text_file,
    ".json": read_text_file,
    ".csv": read_text_file,
    ".html": read_text_file,
    ".pdf": extract_pdf_text,
    ".docx": extract_docx_text,
}

def extract_text_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    fn = SUPPORTED.get(ext, read_text_file)
    try:
        return fn(path) or ""
    except Exception:
        return ""

class Indexer:
    def __init__(self):
        self.vs = VectorStore()
        ensure_dir(os.path.dirname(EVENT_DB) or ".")
        self.conn = sqlite3.connect(EVENT_DB, check_same_thread=False)
        self._ensure_events_table()

    def _ensure_events_table(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                path TEXT,
                timestamp REAL,
                processed INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def fetch_pending_events(self, limit=50):
        c = self.conn.cursor()
        rows = c.execute("SELECT id, event_type, path, timestamp FROM events WHERE processed=0 ORDER BY id LIMIT ?", (limit,)).fetchall()
        return rows

    def mark_event_processed(self, eid):
        c = self.conn.cursor()
        c.execute("UPDATE events SET processed=1 WHERE id=?", (eid,))
        self.conn.commit()

    def process_path(self, path: str, ts: float = None):
        if not os.path.exists(path):
            return False, "missing"
        text = extract_text_for_path(path)
        if not text:
            return False, "no_text"
        sha = sha256_of_text(text)
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            chunks = [text[:CHUNK_SIZE]]
        try:
            embeddings = embed(chunks)
            emb_stack = np.vstack([np.asarray(e, dtype="float32") for e in embeddings])
            file_vec = np.mean(emb_stack, axis=0)
        except Exception as e:
            print("[indexer] embed error:", e)
            # deterministic fallback
            file_vec = np.zeros(EMBED_DIM, dtype="float32")

        summary = text[:800] + ("..." if len(text) > 800 else "")
        timestamp = ts or time.time()
        try:
            # support both 'content' and older 'summary' patterns
            self.vs.upsert(path=path, content=summary, vector=file_vec, timestamp=timestamp, sha256=sha, source="file")
            return True, "indexed"
        except Exception as e:
            return False, f"upsert_error:{e}"

    def process_event_row(self, row):
        eid, et, path, ts = row
        ok, reason = self.process_path(path, ts)
        # mark processed regardless to avoid infinite loops; you may implement retries later
        self.mark_event_processed(eid)
        return ok, reason

    def run_once(self, batch_size=20):
        rows = self.fetch_pending_events(limit=batch_size)
        if not rows:
            return 0
        for r in rows:
            try:
                ok, reason = self.process_event_row(r)
                print(f"[indexer] processed {r[2]} -> {reason}")
            except Exception as e:
                print("[indexer] error:", e)
                self.mark_event_processed(r[0])
        return len(rows)

    def reindex_all_vectors(self):
        c = self.conn.cursor()
        rows = c.execute("SELECT DISTINCT path, timestamp FROM events ORDER BY timestamp DESC").fetchall()
        count = 0
        for path, ts in rows:
            if os.path.exists(path):
                ok, reason = self.process_path(path, ts)
                print(f"[reindex] {path} -> {reason}")
                count += 1
        return count

    def loop(self, poll_interval=POLL_INTERVAL):
        print("[indexer] starting loop (poll interval:", poll_interval, "s)")
        try:
            while True:
                n = self.run_once()
                if n == 0:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[indexer] stopped by user.")
        finally:
            self.conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    idx = Indexer()
    if args.reindex:
        idx.reindex_all_vectors()
    elif args.once:
        idx.run_once()
    else:
        idx.loop()
