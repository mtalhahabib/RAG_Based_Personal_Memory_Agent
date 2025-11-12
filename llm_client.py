# llm_client.py
"""
RAG-ready LLM client supporting Gemini CLI, Ollama, or stub mode.
Handles embeddings and chat generation.
"""
import os
import shutil
import subprocess
import numpy as np

VECTOR_DIM = 3072
GEMINI_CMD = shutil.which("gemini") or shutil.which("gemini.cmd")
OLLAMA_CMD = shutil.which("ollama")

try:
    from google import generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class LLMClient:
    def __init__(self, backend_override=None):
        self.backend = backend_override or os.environ.get("LLM_BACKEND", "gemini")
        self._genai_available = GENAI_AVAILABLE and bool(os.environ.get("GEMINI_API_KEY"))
        if self._genai_available:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    # ----------------------
    # Embeddings
    # ----------------------
    def embed(self, texts):
        vectors = []
        for t in texts:
            vec = None
            if self.backend == "gemini" and self._genai_available:
                try:
                    resp = genai.embed_content(model="gemini-embedding-001", content=t)
                    vec = np.array(resp["embedding"], dtype="float32")
                    if vec.size != VECTOR_DIM:
                        vec = np.pad(vec, (0, VECTOR_DIM - vec.size), mode='wrap')[:VECTOR_DIM]
                except Exception:
                    vec = self._stub_vector(t)
            else:
                vec = self._stub_vector(t)
            vectors.append(vec)
        return vectors

    # ----------------------
    # Chat generation
    # ----------------------
    def generate(self, prompt, max_tokens=512):
        if self.backend == "gemini" and self._genai_available:
            try:
                model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
                resp = model.generate_content(prompt)
                return resp.text.strip() if resp.text else "[Gemini returned empty response]"
            except Exception as e:
                return f"[Gemini Error] {e}"
        elif self.backend == "ollama" and OLLAMA_CMD:
            try:
                result = subprocess.run(
                    ["ollama", "run", os.environ.get("OLLAMA_MODEL", "llama3"), prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                return result.stdout.strip()
            except Exception as e:
                return f"[Ollama Error] {e}"
        else:
            return f"[stub] No LLM connected. Prompt: {prompt}"

    # ----------------------
    # Fallback vector
    # ----------------------
    def _stub_vector(self, text):
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype="uint8").astype("float32")
        if arr.size < VECTOR_DIM:
            arr = np.pad(arr, (0, VECTOR_DIM - arr.size), mode='wrap')
        return arr[:VECTOR_DIM]
