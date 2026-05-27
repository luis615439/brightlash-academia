import os
import time
import hashlib
import sqlite3
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from alchemy_engine import distill_and_convert

# Rutas del Sistema Diamante
DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
KB_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
INPUT_ZONE = "/Volumes/IA_LAB_DAT/INPUT_ZONE"

# Definición de Categorías Semánticas (Robustecida)
CATEGORIES = {
    "LIDERAZGO": ["liderazgo", "gestion", "management", "estrategia"],
    "PNL": ["pnl", "psicologia", "persuasion", "hipnosis", "comunicacion"],
    "MARKETING": ["marketing", "ventas", "branding", "publicidad", "vender"]
}

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class DiamondAutoIngestHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in ['.pdf', '.epub', '.txt', '.md', '.docx']:
            return

        print(f"💎 CENTINELA DETECTÓ NUEVO ACTIVO: {file_path.name}")
        time.sleep(1) # Esperar a que el archivo se escriba completamente
        
        try:
            self.process_file(file_path)
        except Exception as e:
            print(f"❌ Error procesando {file_path.name}: {e}")

    def process_file(self, file_path):
        fhash = get_file_hash(file_path)
        
        # 1. Clasificación Semántica (Simple por ahora, extensible a GraphRAG)
        import unicodedata
        fn_raw = file_path.name.lower()
        fn_lower = "".join(c for c in unicodedata.normalize('NFD', fn_raw) if unicodedata.category(c) != 'Mn')
        topic = "GENERAL"
        
        # Scoring simple por palabras clave (Simulando motor semántico)
        for cat, keywords in CATEGORIES.items():
            if any(k in fn_lower for k in keywords):
                topic = cat
                break
        
        # 2. Autor (Extracción básica del nombre del archivo o metadatos)
        author = "_AUTO_INGEST_" # En producción, aquí iría el motor de extracción
        
        dest_dir = os.path.join(KB_ROOT, topic, author)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, file_path.name)

        # 3. Mover y Registrar
        shutil.move(str(file_path), dest_path)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO files (filename, original_path, current_path, file_hash, category, author, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_path.name, str(file_path), dest_path, fhash, topic, author, topic))
        conn.commit()
        conn.close()

        print(f"✅ ACTIVO INTEGRADO AL DIAMOND VAULT:")
        print(f"   - Archivo: {file_path.name}")
        print(f"   - Hash: {fhash[:10]}...")
        print(f"   - Categoría: {topic}")
        print(f"   - Destino: {dest_path}")
        
        # 4. Alquimia de Contenido
        if topic in ["PNL", "MARKETING", "LIDERAZGO"]:
            distill_and_convert(dest_path, topic)
            
        print(f"📢 PUBLICADO EN EL PORTAL DE CRISTAL (Mockup Sync)")

def start_sentinel():
    event_handler = DiamondAutoIngestHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_ZONE, recursive=False)
    observer.start()
    print(f"🚀 CENTINELA DIAMANTE ACTIVADO EN: {INPUT_ZONE}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_sentinel()
