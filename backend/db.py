# db.py - SQLite setup (6 tables) and tiny insert/query helpers. No ORM.
import sqlite3
from datetime import date

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    model TEXT PRIMARY KEY, input_per_1m REAL, output_per_1m REAL);
CREATE TABLE IF NOT EXISTS requests (
    trace_id TEXT PRIMARY KEY, ts TEXT, query TEXT, answer TEXT, model_used TEXT,
    cache_type TEXT, escalations INTEGER, total_cost REAL, total_ms REAL,
    router_version TEXT);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost REAL, ms REAL, ok INTEGER);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, thumbs INTEGER, ts TEXT);
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_ts TEXT, strategy TEXT, accuracy REAL,
    total_cost REAL, cost_per_correct REAL, savings_pct REAL, avg_ms REAL,
    cache_hit_rate REAL, wrong_hit_rate REAL, tau_small REAL, tau_medium REAL);
CREATE TABLE IF NOT EXISTS router_versions (
    version TEXT PRIMARY KEY, ts TEXT, tune_accuracy REAL, tune_cost REAL,
    is_active INTEGER);
"""


def connect():
    config.DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(prices=None):
    """Create tables; optionally (re)load the prices table from prices.yaml."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        if prices:
            for tier in config.TIERS:
                p = prices[tier]
                conn.execute("INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                             (p["model"], p["input_per_1m"], p["output_per_1m"]))


def insert(table, row):
    """Insert a dict as one row. Returns the new rowid."""
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with connect() as conn:
        cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
        return cur.lastrowid


def query(sql, params=()):
    """Run a SELECT and return a list of dicts."""
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def execute(sql, params=()):
    with connect() as conn:
        conn.execute(sql, params)


def spend_today():
    """Total spend of all requests today (for the daily budget cap)."""
    rows = query("SELECT COALESCE(SUM(total_cost), 0) AS s FROM requests WHERE ts >= ?",
                 (date.today().isoformat(),))
    return rows[0]["s"]
