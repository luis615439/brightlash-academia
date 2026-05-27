import os
import hashlib
import sqlite3
import shutil
from pathlib import Path

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
QUARANTINE_DIR = "/Volumes/IA_LAB_DAT/00_SISTEMA_Y_BASURA/_MIGRACION_OLD_/DUPLICATES_QUARANTINE"

BATCH_LIMIT = 20

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_category(filename):
    # Simple keyword-based categorization
    fn = filename.lower()
    if any(k in fn for k in ['mkt', 'marketing', 'ventas', 'sales', 'vender']):
        return "MARKETING_Y_VENTAS"
    if any(k in fn for k in ['ia', 'ai', 'artificial', 'gpt', 'llm', 'prompt']):
        return "IA_Y_AUTOMATIZACION"
    if any(k in fn for k in ['pnl', 'psicologia', 'persuasion', 'influencia']):
        return "PSICOLOGIA_Y_PNL"
    if any(k in fn for k in ['espiritualidad', 'meditacion', 'zen', 'mindfulness']):
        return "ESPIRITUALIDAD"
    return "GENERAL"

def ingest_file(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        return False
    
    file_hash = get_file_hash(file_path)
    category = get_category(file_path.name)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check for duplicates
    cursor.execute("SELECT current_path FROM files WHERE file_hash = ?", (file_hash,))
    duplicate = cursor.fetchone()
    
    if duplicate:
        print(f"Duplicate detected: {file_path.name}. Moving to quarantine.")
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        shutil.move(str(file_path), os.path.join(QUARANTINE_DIR, file_path.name))
        conn.close()
        return False

    # Manage batches
    cursor.execute("SELECT current_batch_id, file_count FROM batches WHERE category = ?", (category,))
    batch_info = cursor.fetchone()
    
    if not batch_info:
        batch_id = 1
        count = 0
        cursor.execute("INSERT INTO batches (category, current_batch_id, file_count) VALUES (?, ?, ?)", (category, batch_id, count))
    else:
        batch_id, count = batch_info
        if count >= BATCH_LIMIT:
            batch_id += 1
            count = 0
            cursor.execute("UPDATE batches SET current_batch_id = ?, file_count = ? WHERE category = ?", (batch_id, count, category))
    
    # Target directory
    target_dir = os.path.join(KB_ROOT, f"{category}-{batch_id:02d}")
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, file_path.name)
    
    # Move and record
    try:
        shutil.move(str(file_path), target_path)
        cursor.execute('''
        INSERT INTO files (filename, original_path, current_path, file_hash, category, batch_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_path.name, str(file_path), target_path, file_hash, category, batch_id))
        
        cursor.execute("UPDATE batches SET file_count = file_count + 1 WHERE category = ?", (category,))
        
        conn.commit()
        print(f"Ingested: {file_path.name} -> {category}-{batch_id:02d}")
    except Exception as e:
        print(f"Error ingesting {file_path.name}: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    # Test ingestion from a directory
    source_dir = "/Volumes/IA_LAB_DATA/Libreria_Rescate"
    for f in os.listdir(source_dir):
        ingest_file(os.path.join(source_dir, f))
