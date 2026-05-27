import os
import sqlite3
import unicodedata
from datetime import datetime

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/api/diamond_kb.db"
LESSONS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/MICRO_LECCIONES/"

def sync_lessons():
    if not os.path.exists(LESSONS_DIR):
        print(f"❌ Error: El directorio {LESSONS_DIR} no existe.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Asegurar que la tabla existe (aunque ya debería estar por el backend)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            current_path TEXT,
            text_extracted INTEGER DEFAULT 0,
            indexed INTEGER DEFAULT 0,
            summary TEXT,
            niche TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    files = [f for f in os.listdir(LESSONS_DIR) if f.endswith(".md") and not f.startswith(".")]
    
    count = 0
    for f in files:
        f_norm = unicodedata.normalize('NFC', f)
        full_path = os.path.join(LESSONS_DIR, f_norm)
        
        # Intentar insertar si no existe
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO files (filename, current_path, niche, summary)
                VALUES (?, ?, ?, ?)
            """, (f_norm, full_path, "MICRO_LECCIONES", f"Micro-lección destilada: {f_norm.replace('_', ' ')}"))
            
            if cursor.rowcount > 0:
                count += 1
        except Exception as e:
            print(f"⚠️ Error insertando {f_norm}: {e}")
            
    conn.commit()
    conn.close()
    
    print(f"✅ Sincronización completada. Se agregaron {count} nuevas lecciones a la base de datos.")

if __name__ == "__main__":
    sync_lessons()
