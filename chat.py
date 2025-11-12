# chat.py
"""
RAG Chat CLI:
- Pulls context from VectorStore + recent session + local events/git/browser history
- Generates answers via LLMClient
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

from llm_client import LLMClient
from vectorstore import VectorStore
import watcher
import git_watcher
import browser_history

SESSION_MEMORY_SIZE = int(os.environ.get("SESSION_MEMORY_SIZE", 10))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", 4))
MAX_SNIPPET_CHARS = int(os.environ.get("MAX_SNIPPET_CHARS", 800))

# -------------------------
# Session memory
# -------------------------
session_memory = []

def add_to_session(user, assistant):
    session_memory.append({"user": user, "assistant": assistant})
    if len(session_memory) > SESSION_MEMORY_SIZE:
        session_memory.pop(0)

def get_session_context():
    return "\n".join(f"User: {p['user']}\nAssistant: {p['assistant']}" for p in session_memory)

# -------------------------
# Initialize
# -------------------------
llm = LLMClient()
vs = VectorStore()

# -------------------------
# Semantic search helpers
# -------------------------
def semantic_retrieve(query: str, top_k=RAG_TOP_K):
    emb = llm.embed([query])[0]
    results = vs.search(emb, top_k=top_k)
    hits = []
    for score, meta in results:
        path = meta.get("path")
        snippet = meta.get("content", "")
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
        hits.append({"score": float(score), "path": path, "snippet": snippet})
    return hits

def gather_context(query):
    context_docs = []

    # Session memory
    session_ctx = get_session_context()
    if session_ctx:
        context_docs.append({"path": "session_memory", "content": session_ctx})

    # Semantic RAG hits
    hits = semantic_retrieve(query)
    for h in hits:
        context_docs.append({"path": h["path"], "content": h["snippet"]})

    # Recent local file events
    for event in watcher.get_recent_file_events(limit=5):
        context_docs.append({"path": "file_event", "content": event})

    # Recent git commits
    for commit in git_watcher.get_recent_commits(limit=5):
        context_docs.append({"path": "git_commit", "content": commit})

    # Recent browser history
    for entry in browser_history.get_recent_browser_history(limit=5):
        context_docs.append({"path": "browser", "content": entry})

    return context_docs

def generate_reply(user_query: str):
    context_docs = gather_context(user_query)
    context_text = "\n\n".join([f"Source: {d['path']}\n{d['content']}" for d in context_docs])
    prompt = f"Use the following context to answer concisely:\n{context_text}\n\nUser query:\n{user_query}"
    return llm.generate(prompt)

# -------------------------
# CLI
# -------------------------
def main():
    print("Professional AI Memory Assistant (RAG) — type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "bye"):
            print("Goodbye.")
            break

        reply = generate_reply(user)
        print("\nAssistant:\n" + reply + "\n")
        add_to_session(user, reply)

if __name__ == "__main__":
    main()
