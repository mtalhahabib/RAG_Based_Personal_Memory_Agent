# vectorstore.py
import sqlite3
import numpy as np
import os
import json

DB_PATH = os.environ.get("VECTOR_DB", "memory_vectors.db")

def to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype("float32").tobytes()

def from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="float32")

class VectorStore:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY,
            path TEXT,
            source TEXT,
            content TEXT,
            embedding BLOB,
            timestamp REAL,
            sha256 TEXT,
            metadata TEXT
        )
        """)
        self.conn.commit()

    def upsert(self, path, source, content, vector, timestamp, sha256, metadata=None):
        c = self.conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO vectors
        (path, source, content, embedding, timestamp, sha256, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (path, source, content, to_bytes(vector), timestamp, sha256, json.dumps(metadata or {})))
        self.conn.commit()

    def all_embeddings(self):
        c = self.conn.cursor()
        rows = c.execute("SELECT id, path, content, embedding, timestamp FROM vectors").fetchall()
        return [(rid, path, content, from_bytes(emb), ts) for rid, path, content, emb, ts in rows]

    def search(self, query_vector, top_k=5):
        rows = self.all_embeddings()
        if not rows:
            return []
        ids, paths, contents, vecs, ts = zip(*rows)
        mat = np.vstack(vecs)
        q = query_vector.astype("float32")
        q_norm = np.linalg.norm(q) or 1e-12
        mat_norms = np.linalg.norm(mat, axis=1)
        scores = (mat @ q) / np.where(mat_norms * q_norm == 0, 1e-12, mat_norms * q_norm)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), {"id": int(ids[i]), "path": paths[i], "content": contents[i], "timestamp": float(ts[i])}) for i in idx]

    def delete_by_path(self, path):
        c = self.conn.cursor()
        c.execute("DELETE FROM vectors WHERE path=?", (path,))
        self.conn.commit()
