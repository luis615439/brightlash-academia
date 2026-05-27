import os
import hashlib
import sqlite3
import shutil

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
VOLUME_ROOT = "/Volumes/IA_LAB_DAT"
INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"

def get_file_hash_prefix(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except:
        return None

def parse_index():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prefix_map = {}
    for line in lines[6:]:
        if '|' not in line or '---' in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        raw_loc = parts[5].strip('`')
        if not raw_loc or raw_loc == '...': continue
        
        filename_part = raw_loc.split('/')[-1] if '/' in raw_loc else raw_loc
        prefix = filename_part.split('.')[0].replace('...', '')
        
        if not prefix or len(prefix) < 4: continue
        
        prefix_map[prefix] = {
            "title": parts[2],
            "author": parts[3],
            "summary": parts[4]
        }
    return prefix_map

def deep_filesystem_scan():
    print("💎 INICIANDO ESCANEO AGRESIVO (MAXIMA PROFUNDIDAD)...")
    prefix_map = parse_index()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get prefixes already in DB to avoid double work
    cursor.execute("SELECT file_hash FROM files WHERE author IS NOT NULL AND author != ''")
    existing_hashes = {row[0][:len(list(prefix_map.keys())[0])] for row in cursor.fetchall()}
    
    missing_prefixes = {p: meta for p, meta in prefix_map.items() if p not in existing_hashes}
    print(f"Buscando {len(missing_prefixes)} sobrevivientes...")
    
    found_count = 0
    
    # We include EVERYTHING except clearly non-file folders
    SKIP_DIRS = {'.Spotlight-V100', '.Trashes', 'IA_LAB_KNOWLEDGE_BASE'}

    for root, dirs, files in os.walk(VOLUME_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for filename in files:
            if filename.startswith('.'): continue
            
            # AGGRESSIVE CHECK: is the prefix ANYWHERE in the name?
            matched_prefix = None
            for p in missing_prefixes:
                if p in filename:
                    matched_prefix = p
                    break
            
            if not matched_prefix: continue
            
            file_path = os.path.join(root, filename)
            meta = missing_prefixes[matched_prefix]
            
            # Verify with hash
            full_hash = get_file_hash_prefix(file_path)
            if not full_hash or not full_hash.startswith(matched_prefix):
                continue

            print(f"🎯 ENCONTRADO: {filename}")
            
            topic = "GENERAL"
            summary = meta['summary'].lower()
            title = meta['title'].lower()
            if any(k in summary or k in title for k in ['psicología', 'mente', 'pnl', 'emoción']): topic = "PSICOLOGIA"
            elif any(k in summary or k in title for k in ['espiritual', 'meditación', 'zen', 'tao', 'hermetismo']): topic = "ESPIRITUALIDAD"
            elif any(k in summary or k in title for k in ['marketing', 'ventas', 'copywriting', 'negocios']): topic = "MARKETING"
            elif any(k in summary or k in title for k in ['cocina', 'recetas', 'gastronomía']): topic = "GASTRONOMIA"
            
            author = meta['author'] if meta['author'] and meta['author'] != 'Desconocido' else "_DESCONOCIDO_"
            safe_author = author.replace(" ", "_").replace("/", "-")
            dest_dir = os.path.join(KB_ROOT, topic, safe_author)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                shutil.copy2(file_path, dest_path)
                cursor.execute("""
                    INSERT OR REPLACE INTO files (filename, original_path, current_path, file_hash, category, author, topic)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (filename, file_path, dest_path, full_hash, topic, meta['author'], topic))
                
                found_count += 1
                if found_count % 10 == 0:
                    print(f"✅ {found_count} procesados...")
                    conn.commit()
                    
            except Exception as e:
                print(f"❌ Error: {e}")

    conn.commit()
    conn.close()
    print(f"\n✨ REPORTE FINAL AGRESIVO:")
    print(f"Recuperados: {found_count}")

if __name__ == "__main__":
    deep_filesystem_scan()
