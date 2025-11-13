# git_watcher.py
# --------------------------------------------------------------------
# Scans git repositories under WATCH_PATHS and records commit history
# Provides helper to retrieve all commits for RAG context
# --------------------------------------------------------------------
import os
import sqlite3
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

EVENT_DB = os.environ.get("EVENT_DB", "events.db")
GIT_AUTO_DISCOVER = os.environ.get("GIT_AUTO_DISCOVER", "true").lower() == "true"
WATCH_PATHS = [p.strip() for p in os.environ.get("WATCH_PATHS", "").split(",") if p.strip()]
GIT_WATCH_PATHS = [p.strip() for p in os.environ.get("GIT_WATCH_PATHS", "").split(",") if p.strip()]
POLL_INTERVAL = 120  # seconds

# -------------------------
# Database initialization
# -------------------------
def init_db():
    conn = sqlite3.connect(EVENT_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS git_commits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            repo_name TEXT,
            repo_dir TEXT,
            commit_hash TEXT,
            author TEXT,
            date TEXT,
            message TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    print("[git_watcher] Initialized database.")
    return conn

# -------------------------
# Discover git repos recursively
# -------------------------
def discover_git_repos(start_path):
    repos = []
    for root, dirs, _ in os.walk(start_path):
        if ".git" in dirs:
            repos.append(root)
            dirs.remove(".git")  # don't recurse into .git
        for skip in [".venv", "node_modules", "__pycache__", "dist"]:
            if skip in dirs:
                dirs.remove(skip)
    return repos

# -------------------------
# Get commit history for a repo
# -------------------------
def get_commit_history(repo):
    try:
        result = subprocess.check_output(
            ["git", "-C", repo, "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso"],
            text=True
        )
        commits = []
        for line in result.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })
        return commits
    except subprocess.CalledProcessError:
        return []

# -------------------------
# Scan and log new commits
# -------------------------
def scan_and_log(conn):
    c = conn.cursor()
    repos = set(GIT_WATCH_PATHS)
    if GIT_AUTO_DISCOVER:
        for base in WATCH_PATHS:
            repos.update(discover_git_repos(base))
    print(f"[git_watcher] Auto-discover: {GIT_AUTO_DISCOVER}, Watch paths: {WATCH_PATHS}")
    print(f"[git_watcher] Discovered {len(repos)} repos:")
    for r in repos:
        print(f"    {r}")

    for repo in repos:
        commits = get_commit_history(repo)
        for commit in commits:
            c.execute("SELECT 1 FROM git_commits WHERE commit_hash=? AND repo=?", (commit["hash"], repo))
            if not c.fetchone():
                repo_name = os.path.basename(repo.rstrip("\\/"))
                repo_dir = os.path.dirname(repo)
                c.execute("""
                    INSERT INTO git_commits (repo, repo_name, repo_dir, commit_hash, author, date, message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (repo, repo_name, repo_dir, commit["hash"], commit["author"], commit["date"], commit["message"], time.time()))
                conn.commit()

# -------------------------
# Retrieve all repositories and their commits for RAG
# Returns: {repo_path: [commit_str1, commit_str2, ...], ...}
# -------------------------
def get_all_repositories_with_commits():
    conn = sqlite3.connect(EVENT_DB)
    c = conn.cursor()
    repos_with_commits = {}
    # Fetch all commits ordered by repo and timestamp descending
    rows = c.execute("SELECT repo, repo_name, date, message FROM git_commits ORDER BY repo, timestamp DESC").fetchall()
    for repo_path, repo_name, date, message in rows:
        commit_str = f"[{date}] {repo_name}: {message}"
        if repo_path not in repos_with_commits:
            repos_with_commits[repo_path] = []
        repos_with_commits[repo_path].append(commit_str)
    conn.close()
    return repos_with_commits

# -------------------------
# Main watcher loop
# -------------------------
if __name__ == "__main__":
    conn = init_db()
    while True:
        scan_and_log(conn)
        print(f"[git_watcher] Scan complete. Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)
