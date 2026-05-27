import os
import shutil
import sqlite3
from pathlib import Path

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"

def reorganize():
    print("💎 Iniciando Reorganización Física de la Bóveda...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get files that have author and topic
    cursor.execute("SELECT id, filename, current_path, author, topic FROM files WHERE author IS NOT NULL AND topic IS NOT NULL")
    files = cursor.fetchall()
    
    reorganized_count = 0
    errors = []
    
    for fid, filename, current_path, author, topic in files:
        if not current_path or not os.path.exists(current_path):
            print(f"⚠️ Archivo no encontrado: {filename} en {current_path}")
            continue
            
        # Prepare target path: TOPIC / AUTHOR / FILENAME
        # Clean names for folder safety
        safe_topic = topic.replace(" ", "_").upper()
        safe_author = author.replace(" ", "_") if author else "_DESCONOCIDO_"
        
        target_dir = os.path.join(KB_ROOT, safe_topic, safe_author)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, filename)
        
        # Avoid moving if it's already there
        if current_path == target_path:
            continue
            
        try:
            # Check if target already exists (duplicate in target)
            if os.path.exists(target_path):
                print(f"🔄 Duplicado detectado en destino: {filename}. Saltando.")
                continue
                
            shutil.move(current_path, target_path)
            
            # Update DB
            cursor.execute("UPDATE files SET current_path = ? WHERE id = ?", (target_path, fid))
            reorganized_count += 1
            
            if reorganized_count % 50 == 0:
                print(f"✅ Movidos {reorganized_count} activos...")
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error moviendo {filename}: {e}")
            errors.append(filename)

    conn.commit()
    conn.close()
    
    print(f"\n✨ REORGANIZACIÓN COMPLETADA:")
    print(f"Total movidos: {reorganized_count}")
    if errors:
        print(f"Errores en {len(errors)} archivos.")

if __name__ == "__main__":
    reorganize()
