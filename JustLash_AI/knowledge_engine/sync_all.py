import os
import sqlite3
import hashlib
from pathlib import Path
from diamond_ingestor import get_file_hash, get_category, DB_PATH, KB_ROOT

def sync_database():
    print("🚀 Iniciando sincronización de Bóveda...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Obtener archivos ya indexados
    cursor.execute("SELECT current_path FROM files")
    indexed_paths = {row[0] for row in cursor.fetchall()}
    
    new_files = 0
    for root, dirs, files in os.walk(KB_ROOT):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Si no está en la base de datos, lo agregamos
            if file_path not in indexed_paths:
                file_hash = get_file_hash(Path(file_path))
                category = get_category(file)
                
                # Extraer batch_id del nombre de la carpeta (ej. MARKETING_Y_VENTAS-01)
                folder_name = os.path.basename(root)
                try:
                    batch_id = int(folder_name.split('-')[-1])
                except:
                    batch_id = 1
                
                cursor.execute("""
                    INSERT OR IGNORE INTO files (filename, file_hash, category, batch_id, current_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (file, file_hash, category, batch_id, file_path))
                new_files += 1
                print(f"✅ Sincronizado: {file}")
    
    conn.commit()
    conn.close()
    print(f"\n✨ Sincronización completa. Se agregaron {new_files} libros nuevos a la base de datos.")

if __name__ == "__main__":
    sync_database()
