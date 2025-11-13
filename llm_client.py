# llm_client.py
"""
RAG-ready LLM client.
- Uses google-generativeai (preferred if GEMINI_API_KEY present)
- Falls back to Gemini CLI if desired (kept minimal)
- Provides:
    - LLMClient class with embed() and generate()
    - module-level convenience wrappers embed(texts)/chat_completion(...)
"""
import os
import json
import numpy as np
from typing import List, Optional, Dict, Any

# Try to import official google generative ai package (you have it in requirements)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# Config
load_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_DIM = int(os.environ.get("EMBED_DIM", 3072))  # keep large to match Gemini embeddings if available

# Initialize genai if possible
if GENAI_AVAILABLE and load_key:
    try:
        genai.configure(api_key=load_key)
    except Exception:
        # not fatal; will fall back to stubs
        GENAI_AVAILABLE = False

class LLMClient:
    def __init__(self, backend: Optional[str] = None):
        # backend selection is simple: prefer genai when available
        if GENAI_AVAILABLE:
            self.backend = "genai"
        else:
            self.backend = backend or os.environ.get("LLM_BACKEND", "stub").lower()

    # ---------- Embeddings ----------
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        texts: list[str]
        returns: list[np.ndarray] (dtype float32)
        """
        out = []
        for t in texts:
            vec = self._get_embedding(t)
            out.append(vec)
        return out

    def _get_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        if self.backend == "genai" and GENAI_AVAILABLE:
            try:
                # google-generativeai embed helper (best-effort - API wrapper names vary by version)
                # This tries the minimal supported call. If your installed version exposes another function,
                # the try/except will fall back to deterministic stub below.
                resp = genai.embed_content(model="gemini-embedding-001", content=text)
                emb = resp.get("embedding") or resp.get("embeddings") or None
                if emb:
                    arr = np.array(emb, dtype=np.float32)
                    if arr.size != EMBED_DIM:
                        # pad/truncate to EMBED_DIM
                        if arr.size < EMBED_DIM:
                            arr = np.pad(arr, (0, EMBED_DIM - arr.size), mode="wrap")
                        else:
                            arr = arr[:EMBED_DIM]
                    return arr
            except Exception:
                # continue to fallback below
                pass

        # deterministic stub fallback (hash => pseudo-random but deterministic)
        import hashlib
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.RandomState(seed % (2**32))
        return rng.rand(EMBED_DIM).astype("float32")

    # ---------- Generation ----------
    def generate(self, system_prompt: str, user_prompt: str, context: str = "", max_tokens: int = 1024) -> str:
        """
        Compose prompt from system + context + user and generate a reply.
        Signature kept explicit for RAG usage.
        """
        prompt = ""
        if system_prompt:
            prompt += system_prompt.strip() + "\n\n"
        if context:
            prompt += "Context:\n" + context.strip() + "\n\n"
        prompt += "User:\n" + user_prompt.strip()

        if self.backend == "genai" and GENAI_AVAILABLE:
            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                resp = model.generate_content(prompt)
                text = ""
                if hasattr(resp, "text") and resp.text:
                    text = resp.text
                else:
                    # fallback: try dict-like
                    if isinstance(resp, dict):
                        # try to extract candidate text
                        txt = resp.get("candidates", [{}])[0].get("content", [{}])[0].get("text", "")
                        text = txt
                return text.strip() if text else "[Gemini returned empty response]"
            except Exception as e:
                return f"[Gemini Error] {e}"

        # Final fallback: stub
        return f"[stub] No cloud LLM available. Prompt length={len(prompt)}. User asked: {user_prompt[:200]}"

# Convenience top-level wrappers (for older code paths)
_client_singleton: Optional[LLMClient] = None

def get_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton

def embed(texts):
    c = get_client()
    if isinstance(texts, str):
        return c.embed([texts])[0]
    return c.embed(texts)

def generate(system_prompt, user_prompt, context="", max_tokens=1024):
    c = get_client()
    return c.generate(system_prompt=system_prompt, user_prompt=user_prompt, context=context, max_tokens=max_tokens)

if __name__ == "__main__":
    # quick smoke test
    c = get_client()
    print("Backend:", c.backend)
    print("Embedding len:", len(c.embed(["hello"])[0]))
    print("Chat sample:", c.generate("System: be concise", "Say hi", context=""))
