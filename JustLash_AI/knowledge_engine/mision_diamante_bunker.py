import sqlite3
import os
import sys
from pathlib import Path
from alchemy_engine import distill_and_convert

# Configuración de rutas
DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
LESSONS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/MICRO_LECCIONES/"

def get_candidate_files(niches):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Mapeo de nichos a categorías de DB
    category_map = {
        "Negocios": ["MARKETING_Y_VENTAS", "FINANZAS_Y_RIQUEZA", "MARKETING"],
        "Psicologia": ["PSICOLOGIA_Y_PNL", "PSICOLOGIA"],
        "Belleza": [] # Se maneja con LIKE
    }
    
    candidates = []
    
    for niche in niches:
        if niche == "Belleza":
            cursor.execute("""
                SELECT filename, current_path, topic FROM files 
                WHERE category = 'GENERAL' 
                AND (filename LIKE '%Lash%' OR filename LIKE '%Pestaña%' OR filename LIKE '%Belleza%' OR filename LIKE '%Beauty%')
            """)
        else:
            categories = category_map.get(niche, [])
            if not categories: continue
            placeholders = ','.join(['?'] * len(categories))
            cursor.execute(f"SELECT filename, current_path, topic FROM files WHERE category IN ({placeholders})", categories)
        
        candidates.extend(cursor.fetchall())
    
    conn.close()
    return candidates

def run_bunker(niches=["Negocios", "Psicologia", "Belleza"], batch_size=10):
    print(f"🛡️  MODO BUNKER ACTIVADO: Iniciando Rescate para {', '.join(niches)}")
    
    candidates = get_candidate_files(niches)
    print(f"🔍 Encontrados {len(candidates)} candidatos potenciales.")
    
    # Filtrar archivos que ya tienen lección
    existing_lessons = set(os.listdir(LESSONS_DIR))
    to_process = []
    
    for filename, path, topic in candidates:
        lesson_name = f"LECCION_{filename.split('.')[0]}.md"
        # También chequear el formato MICRO_LECCION
        micro_name = f"MICRO_LECCION_{filename.split('.')[0]}.md"
        
        if lesson_name not in existing_lessons and micro_name not in existing_lessons:
            to_process.append((filename, path, topic))
    
    print(f"🎯 {len(to_process)} archivos pendientes de destilación.")
    
    if not to_process:
        print("✅ Todo el conocimiento ya está destilado en el búnker.")
        return

    processed = 0
    for filename, path, topic in to_process[:batch_size]:
        print(f"\n💎 [{processed+1}/{batch_size}] Procesando: {filename}")
        # Determinar tópico si es GENERAL pero de Belleza
        effective_topic = topic
        if topic == "GENERAL" and any(k in filename.lower() for k in ['lash', 'pestaña', 'belleza']):
            effective_topic = "BELLEZA_Y_LASHES"
            
        success = distill_and_convert(path, effective_topic, content_type="leccion")
        if success:
            processed += 1
        else:
            print(f"⚠️ Falló la destilación de {filename}")

    print(f"\n✨ RESCATE COMPLETADO: {processed} nuevas lecciones añadidas a la Bóveda.")

if __name__ == "__main__":
    # Ejecutamos un lote de 20 como solicitó el usuario
    run_bunker(batch_size=20)
