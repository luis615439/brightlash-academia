import sqlite3
import os
import shutil
from pathlib import Path

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"

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

def deep_scan_and_migrate():
    print("💎 Iniciando Escaneo de Contenido Profundo (Hash Prefix Mode)...")
    prefix_map = parse_index()
    print(f"Buscando {len(prefix_map)} sobrevivientes en la base de datos...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    found = 0
    migrated = 0
    errors = 0
    
    for prefix, meta in prefix_map.items():
        # Search for files with this hash prefix
        cursor.execute("SELECT id, filename, current_path, author, topic FROM files WHERE file_hash LIKE ?", (f"{prefix}%",))
        rows = cursor.fetchall()
        
        if not rows:
            # print(f"⚠️ No se encontró rastro de prefix: {prefix}")
            continue
            
        found += 1
        for db_id, filename, current_path, existing_author, existing_topic in rows:
            # If already has author and topic, it's likely already in the right place
            if existing_author and existing_topic:
                continue
            
            if not os.path.exists(current_path):
                # print(f"❌ Archivo perdido físicamente: {filename} en {current_path}")
                errors += 1
                continue

            # Topic inference
            topic = "GENERAL"
            summary = meta['summary'].lower()
            title = meta['title'].lower()
            if any(k in summary or k in title for k in ['psicología', 'mente', 'pnl', 'emoción', 'comportamiento']): topic = "PSICOLOGIA"
            elif any(k in summary or k in title for k in ['espiritual', 'meditación', 'zen', 'tao', 'hermetismo', 'pineal', 'ocultismo']): topic = "ESPIRITUALIDAD"
            elif any(k in summary or k in title for k in ['marketing', 'ventas', 'copywriting', 'negocios', 'persuasión']): topic = "MARKETING"
            elif any(k in summary or k in title for k in ['cocina', 'recetas', 'gastronomía', 'alimento']): topic = "GASTRONOMIA"
            
            # Destination path
            author = meta['author'] if meta['author'] and meta['author'] != 'Desconocido' else "_DESCONOCIDO_"
            safe_author = author.replace(" ", "_").replace("/", "-")
            dest_dir = os.path.join(KB_ROOT, topic, safe_author)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                if os.path.exists(dest_path) and os.path.samefile(current_path, dest_path):
                    # Update DB with metadata even if already there
                    cursor.execute("UPDATE files SET author = ?, topic = ? WHERE id = ?", (meta['author'], topic, db_id))
                    continue

                shutil.copy2(current_path, dest_path)
                
                # Update DB
                cursor.execute("""
                    UPDATE files 
                    SET current_path = ?, author = ?, topic = ?, category = ?
                    WHERE id = ?
                """, (dest_path, meta['author'], topic, topic, db_id))
                
                migrated += 1
                if migrated % 20 == 0:
                    print(f"🔄 Migrados {migrated} activos adicionales...")
                    conn.commit()
                    
            except Exception as e:
                print(f"❌ Error migrando {filename}: {e}")
                errors += 1

    conn.commit()
    conn.close()
    
    print(f"\n✨ REPORTE FINAL ESCANEO PROFUNDO:")
    print(f"Sobrevivientes localizados por Hash: {found}")
    print(f"Activos recién organizados: {migrated}")
    print(f"Errores/Faltantes físicos: {errors}")

if __name__ == "__main__":
    deep_scan_and_migrate()
