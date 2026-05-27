import os
import sqlite3
import shutil
from pathlib import Path
from diamond_ingestor import get_file_hash, get_category, DB_PATH, KB_ROOT

INBOX_ROOT = "/Volumes/IA_LAB_DAT/INBOX"

# Diccionario extendido para clasificación rápida por nombre
NICHE_KEYWORDS = {
    'MARKETING_Y_VENTAS': ['marketing', 'ventas', 'ads', 'seo', 'sem', 'copywriting', 'funnel', 'clientes', 'vender', 'negocios', 'comercial'],
    'IA_Y_AUTOMATIZACION': ['ia', 'ai', 'inteligencia artificial', 'chatgpt', 'openai', 'prompt', 'automation', 'python', 'bot', 'n8n', 'zapier'],
    'PSICOLOGIA_Y_PNL': ['psicologia', 'pnl', 'mente', 'emociones', 'comportamiento', 'terapia', 'ansiedad', 'neuro', 'persuasion', 'influencia'],
    'ESPIRITUALIDAD_Y_BIENESTAR': ['espiritualidad', 'meditacion', 'zen', 'budismo', 'yoga', 'salud', 'dieta', 'ayuno', 'energia', 'chakras', 'universo'],
    'LECTURA_Y_APRENDIZAJE': ['lectura', 'aprender', 'estudiar', 'memoria', 'habitos', 'productividad', 'tiempo', 'metodo', 'curso'],
    'FINANZAS_Y_RIQUEZA': ['dinero', 'riqueza', 'finanzas', 'inversion', 'millonario', 'pobre', 'cripto', 'bitcoin', 'ahorro', 'libertad financiera'],
    'SEDUCCION_Y_RELACIONES': ['seduccion', 'mujer', 'hombres', 'pareja', 'atraccion', 'conquistar', 'amor', 'relaciones'],
}

def mass_categorize():
    print("🚀 Iniciando Clasificación Masiva de 12,000+ libros...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    files_processed = 0
    if not os.path.exists(INBOX_ROOT):
        print("❌ INBOX no encontrada.")
        return

    all_inbox_files = [f for f in os.listdir(INBOX_ROOT) if not f.startswith('.')]
    print(f"📦 Total en INBOX: {len(all_inbox_files)} archivos.")

    for filename in all_inbox_files:
        src_path = os.path.join(INBOX_ROOT, filename)
        
        # Determinar categoría
        category = "GENERAL"
        lower_name = filename.lower()
        for niche, keywords in NICHE_KEYWORDS.items():
            if any(kw in lower_name for kw in keywords):
                category = niche
                break
        
        # Ruta destino (batch simple para velocidad)
        dest_dir = os.path.join(KB_ROOT, f"{category}-01")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        try:
            # Mover archivo
            shutil.move(src_path, dest_path)
            
            # Registrar en DB
            file_hash = get_file_hash(Path(dest_path))
            cursor.execute("""
                INSERT OR REPLACE INTO files (filename, file_hash, category, batch_id, current_path)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, file_hash, category, 1, dest_path))
            
            files_processed += 1
            if files_processed % 100 == 0:
                print(f"✅ Procesados {files_processed}...")
                conn.commit()
        except Exception as e:
            print(f"⚠️ Error procesando {filename}: {e}")

    conn.commit()
    conn.close()
    print(f"✨ ¡ÉXITO! Se clasificaron y movieron {files_processed} libros a sus nuevos nichos.")

if __name__ == "__main__":
    mass_categorize()
