import sqlite3
import os

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Files table: tracks all unique files
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_path TEXT,
        current_path TEXT,
        file_hash TEXT UNIQUE NOT NULL,
        category TEXT,
        batch_id INTEGER,
        indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Categories table: tracks batch IDs and counts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS batches (
        category TEXT PRIMARY KEY,
        current_batch_id INTEGER DEFAULT 1,
        file_count INTEGER DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
