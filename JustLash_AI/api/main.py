import os
import sys
import sqlite3
import shutil
import hashlib
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from knowledge_engine.knowledge_bridge import KnowledgeBridge
from knowledge_engine.diamond_ingestor import get_file_hash, get_category, DB_PATH, KB_ROOT, QUARANTINE_DIR, BATCH_LIMIT
from agent_router import AgentRouter

app = FastAPI(title="Diamond Vault API")

# Habilitar CORS para el frontend de Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar puente de conocimiento
kb = KnowledgeBridge()

# --- Modelos Pydantic ---

class SearchQuery(BaseModel):
    text: str
    top_k: int = 3

class SearchResult(BaseModel):
    text: str
    source: str

class Stats(BaseModel):
    total_files: int
    total_chunks: int
    categories: List[dict]

# --- Utilidades de IA Categorizer ---

def ai_get_category(filename: str, content_preview: str) -> str:
    """Usa IA para determinar el nicho si no hay match por nombre."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "GENERAL"
    
    prompt = f"""Analiza este contenido de un libro y determina su nicho específico. 
Si encaja en alguno de estos: [MARKETING_Y_VENTAS, IA_Y_AUTOMATIZACION, PSICOLOGIA_Y_PNL, ESPIRITUALIDAD], responde SOLO con el nombre de la categoría.
Si NO encaja, propón un NUEVO nicho de máximo 2 palabras en MAYÚSCULAS (ej. 'BIOHACKING', 'FINANZAS').

Nombre del archivo: {filename}
Contenido (primeros 1000 caracteres): {content_preview}

Respuesta (SOLO la categoría):"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        category = response.json()['choices'][0]['message']['content'].strip().upper()
        # Limpieza básica
        return category.replace(" ", "_").replace('"', '').replace("'", "")
    except Exception as e:
        print(f"Error en AI Categorizer: {e}")
        return "GENERAL"

# --- Endpoints ---

@app.get("/api/stats", response_model=Stats)
def get_stats():
    # Conteo físico confiable
    physical_count = 0
    for path in [KB_ROOT, "/Volumes/IA_LAB_DAT/INBOX"]:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                physical_count += sum(1 for f in files if not f.startswith('.'))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    indexed_files = cursor.fetchone()[0]
    
    cursor.execute("SELECT category, COUNT(*) FROM files GROUP BY category")
    categories = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return Stats(
        total_files=physical_count,
        total_chunks=indexed_files, # Reutilizamos este campo para mostrar los indexados
        categories=categories
    )

@app.get("/api/files")
def list_files(category: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if category:
        cursor.execute("SELECT filename, category, batch_id, current_path FROM files WHERE category = ?", (category,))
    else:
        cursor.execute("SELECT filename, category, batch_id, current_path FROM files")
    
    files = [{"filename": row[0], "category": row[1], "batch_id": row[2], "path": row[3]} for row in cursor.fetchall()]
    conn.close()
    return files

@app.get("/api/files/content")
def get_file_content(filename: str):
    print(f"DEBUG: Content request for: '{filename}'")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT current_path FROM files WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        
        if not row:
            print(f"DEBUG: Exact match failed for '{filename}', trying LIKE...")
            cursor.execute("SELECT current_path, filename FROM files WHERE filename LIKE ?", (f"%{filename}%",))
            row = cursor.fetchone()
            if row:
                print(f"DEBUG: Found similar file: '{row[1]}'")
        
        conn.close()
        
        if not row:
            print(f"ERROR: File '{filename}' not found in DB")
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        file_path = row[0]
        if not os.path.exists(file_path):
            print(f"ERROR: Physical file missing at {file_path}")
            raise HTTPException(status_code=404, detail="El archivo físico no existe en el disco")
    except Exception as e:
        print(f"ERROR in get_file_content: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    
    from knowledge_engine.diamond_indexer import extract_text
    content = extract_text(file_path)
    # Devolver solo los primeros 5000 caracteres para el preview
    return {"filename": filename, "content": content[:5000], "full_length": len(content)}

@app.get("/api/files/download")
def download_file(filename: str):
    from fastapi.responses import FileResponse
    print(f"DEBUG: Download request for: '{filename}'")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT current_path FROM files WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        
        if not row:
            cursor.execute("SELECT current_path FROM files WHERE filename LIKE ?", (f"%{filename}%",))
            row = cursor.fetchone()
            
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        file_path = row[0]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="El archivo físico no existe")
    except Exception as e:
        print(f"ERROR in download_file: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    
    return FileResponse(path=file_path, filename=filename)

@app.get("/api/search")
def search_get(q: str = ""):
    return search(SearchQuery(text=q, top_k=5))

@app.post("/api/search")
def search(query: SearchQuery):
    print(f"DEBUG: Search request received: {query.text}")
    try:
        # Modificamos KnowledgeBridge para devolver objetos, no solo strings
        raw_results = kb.get_raw_results(query.text, top_k=query.top_k)
        
        sources = list(set([os.path.basename(r['source']) for r in raw_results]))
        
        # Construir respuesta formateada
        formatted_response = "💡 Sabiduría Extraída de tu Bóveda:\n\n"
        for r in raw_results:
            filename = os.path.basename(r['source'])
            clean_chunk = "".join(char for char in r['text'] if char.isprintable() or char in "\n\r\t")
            formatted_response += f"📖 FUENTE: {filename}\n"
            formatted_response += f"└─ {clean_chunk}\n\n"
            
        return {
            "response": formatted_response,
            "sources": sources,
            "raw": raw_results
        }
    except Exception as e:
        print(f"ERROR in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    # Guardar temporalmente
    temp_path = Path(f"/tmp/{file.filename}")
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file_hash = get_file_hash(temp_path)
    
    # Check duplicate
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM files WHERE file_hash = ?", (file_hash,))
    if cursor.fetchone():
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        shutil.move(str(temp_path), os.path.join(QUARANTINE_DIR, file.filename))
        conn.close()
        return {"status": "duplicate", "message": "Archivo duplicado enviado a cuarentena."}

    # Categorización Inteligente
    category = get_category(file.filename)
    if category == "GENERAL":
        # Extraer preview para la IA
        from knowledge_engine.diamond_indexer import extract_text
        preview = extract_text(str(temp_path))[:1000]
        category = ai_get_category(file.filename, preview)

    # Ingestión (Reutilizamos la lógica de diamond_ingestor pero adaptada)
    # [Lógica de batching y movimiento similar a diamond_ingestor.py]
    # Por brevedad, aquí llamamos a una versión modular de la ingesta
    from knowledge_engine.diamond_ingestor import ingest_file
    # (En una versión real, refactorizaríamos diamond_ingestor para ser importable limpiamente)
    
    # Por ahora, simplemente movemos el archivo al lugar que determine diamond_ingestor
    # Nota: ingest_file ya hace el movimiento y el registro en DB.
    
    # Para este MVP, vamos a llamar a la función ingest_file directamente
    # Pero primero devolvemos el archivo a una ruta que diamond_ingestor reconozca
    final_temp = Path(f"/Volumes/IA_LAB_DATA/Libreria_Rescate/{file.filename}")
    shutil.move(str(temp_path), str(final_temp))
    
    # Importamos y ejecutamos
    import importlib
    ingest_mod = importlib.import_module("knowledge_engine.diamond_ingestor")
    ingest_mod.ingest_file(str(final_temp))
    
    conn.close()
    return {"status": "success", "category": category, "message": "Archivo procesado e indexado."}


# --- Integración con AgentRouter de JustLash ---

agent_router = AgentRouter(dry_run=os.getenv("DRY_RUN", "False").lower() == "true")

class ChatRequest(BaseModel):
    lead_id: str
    message: str

class ResetRequest(BaseModel):
    lead_id: str

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """
    Endpoint para procesar los mensajes de leads a través del AgentRouter.
    Ideal para integraciones con webhooks de n8n, WhatsApp o Telegram.
    """
    try:
        response = agent_router.respond(
            lead_id=request.lead_id,
            message=request.message
        )
        return {
            "lead_id": response.lead_id,
            "message": response.message,
            "agent_name": response.agent_name,
            "agent_type": response.agent_type,
            "state_before": response.state_before,
            "state_after": response.state_after,
            "segment": response.segment,
            "transition_occurred": response.transition_occurred,
            "dry_run": response.dry_run,
            "model_used": response.model_used,
            "tokens_used": response.tokens_used,
            "timestamp": response.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en AgentRouter: {str(e)}")

@app.post("/api/chat/reset")
def reset_lead_endpoint(request: ResetRequest):
    """
    Endpoint para reiniciar el estado e historial de un lead.
    """
    success = agent_router.reset_lead(request.lead_id)
    if success:
        return {"status": "success", "message": f"Lead {request.lead_id} reiniciado correctamente."}
    else:
        return {"status": "not_found", "message": f"No se pudo encontrar o reiniciar el lead {request.lead_id}."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
