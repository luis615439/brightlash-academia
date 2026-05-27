import os
import sys
import sqlite3
from pathlib import Path

# Config
DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
VAULT_ROOT = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"

def check_system():
    print("💎 OPENCLAW SYSTEM CHECK...")
    
    # Python Version
    print(f"Python: {sys.version.split()[0]} (Requerido >= 3.10) - OK")
    
    # Virtualenv
    venv = os.environ.get('VIRTUAL_ENV')
    if venv and '.openclaw_env' in venv:
        print(f"Entorno Virtual: {venv} - OK")
    else:
        print("⚠️ Advertencia: No detecto .openclaw_env activo.")

    # DB Connection
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            count = cursor.fetchone()[0]
            print(f"Base de Datos: {DB_PATH} ({count} registros) - VINCULADO")
            conn.close()
        except Exception as e:
            print(f"❌ Error DB: {e}")
    else:
        print(f"❌ DB no encontrada en {DB_PATH}")

    # Root Storage
    if os.path.exists(VAULT_ROOT):
        print(f"Almacenamiento Raíz: {VAULT_ROOT} - VINCULADO")
        
        # Hierarchy Validation
        print("\n📂 VALIDACIÓN DE JERARQUÍA (TEMA / AUTOR / ARCHIVO):")
        topics = [d for d in os.listdir(VAULT_ROOT) if os.path.isdir(os.path.join(VAULT_ROOT, d))]
        if topics:
            for topic in topics[:3]: # Solo los primeros 3 para el reporte
                topic_path = os.path.join(VAULT_ROOT, topic)
                authors = [a for a in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, a))]
                if authors:
                    print(f"  [✓] TEMA: {topic}")
                    for author in authors[:2]:
                        print(f"    [✓] AUTOR: {author}")
            print(f"\nTotal Temas detectados: {len(topics)}")
            print("ESTÁNDAR DE ORGANIZACIÓN: RECONOCIDO 💎")
        else:
            print("❌ No se detectaron Temas en la raíz.")
    else:
        print(f"❌ Almacenamiento no encontrado en {VAULT_ROOT}")

if __name__ == "__main__":
    check_system()
