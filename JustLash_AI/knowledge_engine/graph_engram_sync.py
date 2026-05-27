import json
import sqlite3
import os

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
GRAPH_ANALYSIS = "/Volumes/IA_LAB_DAT/BIBLIOTECA_IDENTIFICADA/graphify-out/.graphify_analysis.json"
ASSETS_MAP = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/assets_map_draft.json"

def sync():
    # 1. Load data
    with open(ASSETS_MAP, 'r') as f:
        assets = json.load(f)
    
    with open(GRAPH_ANALYSIS, 'r') as f:
        graph = json.load(f)
    
    communities = graph.get("communities", {})
    gods = graph.get("gods", [])
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 2. Process assets and update DB
    print("Updating database with Author and Topic...")
    for asset in assets:
        # Infer topic from summary/title (simplified for now, can be improved)
        topic = "GENERAL"
        lower_summary = asset['summary'].lower()
        if "psicología" in lower_summary or "mente" in lower_summary:
            topic = "PSICOLOGIA"
        elif "espiritual" in lower_summary or "meditación" in lower_summary:
            topic = "ESPIRITUALIDAD"
        elif "ventas" in lower_summary or "marketing" in lower_summary:
            topic = "MARKETING"
        elif "recetas" in lower_summary or "cocina" in lower_summary:
            topic = "GASTRONOMIA"
        
        # Clean location to get filename
        loc = asset['location'].strip('`').split('/')[-1]
        
        cursor.execute("""
            UPDATE files SET author = ?, topic = ? 
            WHERE filename LIKE ? OR current_path LIKE ?
        """, (asset['author'], topic, f"%{loc}%", f"%{loc}%"))

    conn.commit()
    print(f"Updated {len(assets)} metadata entries in DB.")
    
    # 3. Report major findings to Engram (Mocked for script, I will call the tool manually)
    # This logic is just to prepare the summary
    findings = []
    for god in gods[:5]:
        findings.append(f"- God Node: {god['label']} (Degree: {god['degree']})")
    
    print("\nGraphRAG Scan Findings:")
    for f in findings:
        print(f)

    conn.close()

if __name__ == "__main__":
    sync()
