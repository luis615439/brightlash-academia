import sqlite3
import os
from alchemy_engine import distill_and_convert

# Configuración
DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
LESSONS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/MICRO_LECCIONES/"
RESOURCES_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/RECURSOS_VENTAS/"

os.makedirs(LESSONS_DIR, exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)

def get_relevant_assets():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Categorías clave relacionadas con Ventas, Persuasión, PNL, Inteligencia Emocional
    categories = ['MARKETING', 'LIDERAZGO', 'PNL', 'VENTAS', 'PSICOLOGIA'] 
    placeholders = ', '.join(['?'] * len(categories))
    query = f"SELECT current_path, category FROM files WHERE category IN ({placeholders}) ORDER BY RANDOM() LIMIT 416"
    cursor.execute(query, categories)
    assets = cursor.fetchall()
    conn.close()
    return assets

def activate_distiller():
    print("💎 ACTIVANDO DESTILADOR MAESTRO - ESTÁNDAR DIAMANTE")
    assets = get_relevant_assets()
    
    # 1. Generar 10 Micro-Lecciones diarias
    print("\n📖 GENERANDO 10 MICRO-LECCIONES DE ALTO IMPACTO...")
    for i in range(min(10, len(assets))):
        asset_path, category = assets[i]
        distill_and_convert(asset_path, category, content_type="leccion")
    
    # 2. Refinar Guiones de Remarketing
    print("\n🧠 REFINANDO GUIONES DE REMARKETING (ARQUITECTAS DE LA BELLEZA)...")
    # Usamos un activo potente de PNL o Ventas para el guion
    if assets:
        asset_path, category = assets[0] # Usar el primer activo aleatorio relevante
        distill_and_convert(asset_path, "CIERRE DE VENTAS AVANZADO", content_type="guion")

    print("\n✅ PROCESO DE DESTILACIÓN COMPLETADO.")

if __name__ == "__main__":
    activate_distiller()
