import sqlite3
import os
import shutil
import hashlib
from pathlib import Path

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
IDENTIFIED_ROOT = "/Volumes/IA_LAB_DAT/BIBLIOTECA_IDENTIFICADA"
INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def parse_index():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prefix_map = {}
    for line in lines[6:]:
        if '|' not in line or '---' in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6:
            continue
        
        raw_loc = parts[5].strip('`')
        if not raw_loc or raw_loc == '...': continue
        
        # Extract prefix from "Folder/Prefix...ext"
        filename_part = raw_loc.split('/')[-1] if '/' in raw_loc else raw_loc
        prefix = filename_part.split('.')[0].replace('...', '')
        
        if not prefix or len(prefix) < 4: continue
        
        prefix_map[prefix] = {
            "title": parts[2],
            "author": parts[3],
            "summary": parts[4]
        }
    return prefix_map

def execute():
    print("💎 Inciando Ejecución Final Misión Diamante...")
    prefix_map = parse_index()
    print(f"Loaded {len(prefix_map)} unique prefixes from index.")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    processed = 0
    duplicates = 0
    
    # We'll search in the IDENTIFIED_ROOT recursively
    # And also in the OLD KNOWLEDGE BASE just in case
    SEARCH_PATHS = [IDENTIFIED_ROOT, KB_ROOT]
    
    for search_root in SEARCH_PATHS:
        print(f"Scanning {search_root}...")
        for root, dirs, files in os.walk(search_root):
            # Optimization: skip the target folders we are creating
            if "IA_LAB_KNOWLEDGE_BASE/PSICOLOGIA" in root or "IA_LAB_KNOWLEDGE_BASE/ESPIRITUALIDAD" in root:
                continue

            for filename in files:
                if filename.startswith('.'): continue
                
                # Match prefix
                matched_metadata = None
                for prefix, meta in prefix_map.items():
                    if filename.startswith(prefix):
                        matched_metadata = meta
                        break
                
                if not matched_metadata: continue

                file_path = os.path.join(root, filename)
                
                # Topic inference
                topic = "GENERAL"
                summary = matched_metadata['summary'].lower()
                title = matched_metadata['title'].lower()
                if any(k in summary or k in title for k in ['psicología', 'mente', 'pnl', 'emoción', 'comportamiento']): topic = "PSICOLOGIA"
                elif any(k in summary or k in title for k in ['espiritual', 'meditación', 'zen', 'tao', 'hermetismo', 'pineal', 'ocultismo']): topic = "ESPIRITUALIDAD"
                elif any(k in summary or k in title for k in ['marketing', 'ventas', 'copywriting', 'negocios', 'persuasión']): topic = "MARKETING"
                elif any(k in summary or k in title for k in ['cocina', 'recetas', 'gastronomía', 'alimento']): topic = "GASTRONOMIA"
                
                # Destination path
                author = matched_metadata['author'] if matched_metadata['author'] and matched_metadata['author'] != 'Desconocido' else "_DESCONOCIDO_"
                safe_author = author.replace(" ", "_").replace("/", "-")
                dest_dir = os.path.join(KB_ROOT, topic, safe_author)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                
                try:
                    # Move logic (copy for now to be safe)
                    if os.path.exists(dest_path):
                        # print(f"⏩ Ya existe: {filename}")
                        continue

                    shutil.copy2(file_path, dest_path)
                    
                    # Update DB
                    fhash = get_file_hash(file_path)
                    cursor.execute("""
                        INSERT OR REPLACE INTO files (filename, original_path, current_path, file_hash, category, author, topic)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (filename, file_path, dest_path, fhash, topic, matched_metadata['author'], topic))
                    
                    processed += 1
                    if processed % 50 == 0:
                        print(f"✅ Procesados {processed} activos...")
                        conn.commit()
                        
                except Exception as e:
                    print(f"❌ Error procesando {filename}: {e}")

    conn.commit()
    conn.close()
    print(f"\n✨ MISIÓN CUMPLIDA:")
    print(f"Activos procesados y organizados: {processed}")

if __name__ == "__main__":
    execute()
