import os
import shutil
import sqlite3
import difflib
import hashlib
from pathlib import Path

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
LODO_ROOT = "/Volumes/IA_LAB_DAT/REVISION_FINAL_LODO"
VOLUME_ROOT = "/Volumes/IA_LAB_DAT"
INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_ratio(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def parse_index():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    survivors = []
    for line in lines[6:]:
        if '|' not in line or '---' in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        survivors.append({
            "id": parts[1],
            "title": parts[2],
            "author": parts[3],
            "summary": parts[4]
        })
    return survivors

def clean_and_rescue():
    print("💎 INICIANDO BARRIDO FINAL (FIXED): FUZZY MATCHING + AISLAMIENTO DE LODO...")
    survivors = parse_index()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT filename FROM files WHERE author IS NOT NULL AND author != ''")
    organized_files = {row[0] for row in cursor.fetchall()}
    
    missing_survivors = [s for s in survivors if s['title'] not in organized_files]
    print(f"Buscando {len(missing_survivors)} sobrevivientes por similitud...")
    
    os.makedirs(LODO_ROOT, exist_ok=True)
    
    rescued_count = 0
    mud_count = 0
    
    SWEEP_DIRS = ['BIBLIOTECA_IDENTIFICADA', 'Nichos', '00_SISTEMA_Y_BASURA', 'GENERAL-01']
    
    for sweep_dir in SWEEP_DIRS:
        target_path = os.path.join(VOLUME_ROOT, sweep_dir)
        if not os.path.exists(target_path): continue
        
        print(f"🧹 Barriendo: {sweep_dir}...")
        for root, dirs, files in os.walk(target_path):
            for filename in files:
                if filename.startswith('.'): continue
                file_path = os.path.join(root, filename)
                
                if filename in organized_files: continue
                
                clean_name = filename.split('.')[0]
                if '_' in clean_name and len(clean_name.split('_')[0]) == 10:
                    clean_name = '_'.join(clean_name.split('_')[1:])

                best_match = None
                max_ratio = 0
                for s in missing_survivors:
                    ratio = get_ratio(clean_name, s['title'])
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_match = s
                
                if max_ratio > 0.85:
                    print(f"🎯 RESCATE FUZZY ({int(max_ratio*100)}%): {filename} -> {best_match['title']}")
                    
                    topic = "GENERAL"
                    sum_lower = best_match['summary'].lower()
                    if any(k in sum_lower for k in ['psicología', 'pnl', 'mente']): topic = "PSICOLOGIA"
                    elif any(k in sum_lower for k in ['espiritual', 'meditación', 'ocultismo']): topic = "ESPIRITUALIDAD"
                    elif any(k in sum_lower for k in ['marketing', 'ventas']): topic = "MARKETING"
                    
                    author = best_match['author'] if best_match['author'] != 'Desconocido' else "_DESCONOCIDO_"
                    dest_dir = os.path.join(KB_ROOT, topic, author.replace(" ", "_"))
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, filename)
                    
                    try:
                        fhash = get_file_hash(file_path)
                        shutil.move(file_path, dest_path)
                        cursor.execute("""
                            INSERT OR REPLACE INTO files (filename, original_path, current_path, file_hash, category, author, topic)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (filename, file_path, dest_path, fhash, topic, best_match['author'], topic))
                        rescued_count += 1
                        organized_files.add(filename)
                    except Exception as e:
                        print(f"❌ Error rescate {filename}: {e}")
                else:
                    try:
                        dest_lodo = os.path.join(LODO_ROOT, filename)
                        shutil.move(file_path, dest_lodo)
                        mud_count += 1
                    except: pass

    conn.commit()
    conn.close()
    print(f"\n✨ REPORTE DE BARRIDO FINAL:")
    print(f"Activos rescatados: {rescued_count}")
    print(f"Archivos aislados en LODO: {mud_count}")

if __name__ == "__main__":
    clean_and_rescue()
