# vectorstore.py
import sqlite3
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional

DB_PATH = os.environ.get("VECTOR_DB", "memory_vectors.db")

def to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype("float32").tobytes()

def from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="float32")

class VectorStore:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            source TEXT,
            content TEXT,
            embedding BLOB,
            timestamp REAL,
            sha256 TEXT,
            metadata TEXT
        )
        """)
        self.conn.commit()

    def upsert(self, *args, **kwargs):
        """
        Flexible upsert:
            upsert(path=..., content=..., vector=..., timestamp=..., sha256=..., source=..., metadata={})
            or older callers may pass summary kw.
        """
        path = kwargs.get("path") or (args[0] if len(args) > 0 else None)
        # Accept either 'content' or 'summary'
        content = kwargs.get("content") or kwargs.get("summary") or ""
        vector = kwargs.get("vector")
        timestamp = kwargs.get("timestamp", None) or time_now()
        sha256 = kwargs.get("sha256", "")
        source = kwargs.get("source", "file")
        metadata = kwargs.get("metadata", {}) or {}

        if path is None or vector is None:
            raise ValueError("VectorStore.upsert requires path and vector")

        c = self.conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO vectors
        (path, source, content, embedding, timestamp, sha256, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (path, source, content, to_bytes(np.asarray(vector, dtype="float32")), timestamp, sha256, json.dumps(metadata)))
        self.conn.commit()

    def all_embeddings(self) -> List[Tuple[int, str, str, np.ndarray, float]]:
        c = self.conn.cursor()
        rows = c.execute("SELECT id, path, content, embedding, timestamp FROM vectors").fetchall()
        return [(rid, path, content, from_bytes(emb), ts) for rid, path, content, emb, ts in rows]

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        rows = self.all_embeddings()
        if not rows:
            return []
        ids, paths, contents, vecs, ts = zip(*rows)
        mat = np.vstack(vecs)
        q = np.asarray(query_vector, dtype="float32")
        q_norm = np.linalg.norm(q) or 1e-12
        mat_norms = np.linalg.norm(mat, axis=1)
        denom = mat_norms * q_norm
        scores = (mat @ q) / np.where(denom == 0, 1e-12, denom)
        idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in idx:
            out.append((float(scores[i]), {"id": int(ids[i]), "path": paths[i], "content": contents[i], "timestamp": float(ts[i])}))
        return out

    def delete_by_path(self, path: str):
        c = self.conn.cursor()
        c.execute("DELETE FROM vectors WHERE path=?", (path,))
        self.conn.commit()

def time_now():
    import time
    return time.time()
