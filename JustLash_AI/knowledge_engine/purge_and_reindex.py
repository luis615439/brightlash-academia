import os
import sqlite3
import shutil
import sys

# Añadir el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
INDEX_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_index.faiss"
METADATA_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_metadata.pkl"

def purge_and_reindex():
    print("💎 PURGA Y REINDEXACIÓN DE BASE DE CONOCIMIENTO DIAMANTE 💎")
    print("=" * 60)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Base de datos no encontrada en: {DB_PATH}")
        return

    # 1. Escanear salud
    print("🔍 Escaneando archivos en base de datos para identificar sanos y corruptos...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, current_path, filename FROM files")
    all_files = cursor.fetchall()
    
    healthy_ids = []
    corrupted_ids = []
    
    for file_id, path, filename in all_files:
        if not os.path.exists(path):
            corrupted_ids.append(file_id)
            continue
        try:
            file_size = os.path.getsize(path)
            if file_size == 0:
                corrupted_ids.append(file_id)
                continue
            with open(path, 'rb') as f:
                head = f.read(100)
                if all(b == 0 for b in head):
                    corrupted_ids.append(file_id)
                else:
                    healthy_ids.append(file_id)
        except:
            corrupted_ids.append(file_id)
            
    print(f"-> Archivos catalogados: {len(all_files)}")
    print(f"-> Archivos detectados como SANOS: {len(healthy_ids)}")
    print(f"-> Archivos detectados como CORRUPTOS/VACÍOS: {len(corrupted_ids)}")
    
    if not healthy_ids:
        print("❌ ERROR: No se detectó ningún archivo sano. Abortando purga para no vaciar la base de datos.")
        conn.close()
        return

    # 2. Eliminar corruptos de la base de datos
    print("\n🗑️ Purgando archivos corruptos de la base de datos SQLite...")
    # SQL in operator has limits, so we batch it
    batch_size = 500
    for i in range(0, len(corrupted_ids), batch_size):
        batch = corrupted_ids[i:i+batch_size]
        cursor.execute(f"DELETE FROM files WHERE id IN ({','.join(['?']*len(batch))})", batch)
    
    # 3. Reiniciar indexación para archivos sanos
    print("🔄 Reiniciando estado de indexación de archivos sanos...")
    cursor.execute("UPDATE files SET text_extracted = 0, vector_indexed = 0")
    
    conn.commit()
    conn.close()
    print("✅ Base de datos SQLite depurada con éxito.")

    # 4. Eliminar archivos de índice anteriores
    print("\n🧹 Eliminando índices vectoriales corruptos anteriores...")
    for p in [INDEX_PATH, METADATA_PATH]:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"   - Eliminado: {os.path.basename(p)}")
            except Exception as e:
                print(f"   - Error al eliminar {os.path.basename(p)}: {e}")
        else:
            print(f"   - No existía: {os.path.basename(p)}")

    # 5. Ejecutar indexador oficial
    print("\n🚀 Iniciando indexación de los archivos sanos desde cero...")
    try:
        from diamond_indexer import index_files
        index_files()
        print("🎉 ¡PROCESO COMPLETADO CON ÉXITO! La base de conocimiento está 100% limpia.")
    except Exception as e:
        print(f"❌ Error durante la reindexación: {e}")

if __name__ == "__main__":
    purge_and_reindex()
