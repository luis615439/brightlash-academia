import os
import sys
import sqlite3
import shutil
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# Configuración de rutas para importar módulos locales
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Importaciones reales de tu infraestructura JustLash
from agent_router import AgentRouter
from knowledge_engine.knowledge_bridge import KnowledgeBridge
from knowledge_engine.diamond_ingestor import DB_PATH, KB_ROOT

# Inicialización de la API de Producción
app = FastAPI(
    title="JustLash VIP Diamond API",
    description="Backend unificado: Bóveda de Libros Semántica + Enjambre de Selección Psicológica.",
    version="2.1.0"
)

# Habilitar CORS para el frontend de Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización de los motores reales en memoria
knowledge_bridge = KnowledgeBridge()
agent_router = AgentRouter(dry_run=os.getenv("DRY_RUN", "False").lower() == "true")

# --- MODELOS DE CONTRATOS DE DATOS (Pydantic) ---
class ChatPayload(BaseModel):
    lead_id: str
    message: str
    phone: Optional[str] = None
    current_status: Optional[str] = "qualifying"

class ResetPayload(BaseModel):
    lead_id: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "JustLash VIP Diamond API v2.1.0",
        "engine": "AgentRouter (Open Swarm) + KnowledgeBridge (SQLite Chunks)"
    }


# =================================================================
# 1. ENDPOINTS DEL ENJAMBRE DE AGENTES (Conexión Real con n8n)
# =================================================================

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    """
    Endpoint principal de producción. Recibe el impacto desde n8n (Telegram/WhatsApp),
    lo procesa a través del AgentRouter real y devuelve la respuesta estructurada.
    """
    try:
        if not payload.lead_id or not payload.message:
            raise HTTPException(status_code=400, detail="Faltan parámetros obligatorios: lead_id o message")
        
        # Orquestación real con persistencia en conversations.json
        # Despierta a Sofía, Mariana o Valeria según la fase del lead
        response = agent_router.respond(lead_id=payload.lead_id, message=payload.message)
        
        # Retorna el contrato exacto que tu Switch de n8n sabe leer
        return {
            "status": "success",
            "state_after": response.state_after,
            "response": response.message,
            "tokens_used": response.tokens_used,
            "metadata": {
                "lead_id": response.lead_id,
                "agent_name": response.agent_name,
                "agent_type": response.agent_type,
                "state_before": response.state_before,
                "segment": response.segment,
                "transition_occurred": response.transition_occurred,
                "dry_run": response.dry_run,
                "model_used": response.model_used,
                "timestamp": response.timestamp
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en el runtime del AgentRouter: {str(e)}")

@app.post("/api/chat/reset")
async def chat_reset_endpoint(payload: ResetPayload):
    """
    Endpoint de mantenimiento del Laboratorio. 
    Limpia el historial de un lead para reiniciar simulaciones desde cero.
    """
    try:
        success = agent_router.reset_lead(payload.lead_id)
        if success:
            return {
                "status": "success",
                "message": f"Historial y estados del lead {payload.lead_id} purgados correctamente."
            }
        else:
            return {
                "status": "not_found",
                "message": f"No se pudo encontrar o reiniciar el lead {payload.lead_id}."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resetear el lead: {str(e)}")


# =================================================================
# 2. ENDPOINTS DE LA DIAMOND VAULT (Bóveda de Libros e Ingesta)
# =================================================================

@app.get("/api/stats")
async def get_vault_stats():
    """Devuelve el conteo físico e indexado de libros en Next.js."""
    try:
        # Conteo físico confiable
        physical_count = 0
        for path in [KB_ROOT, "/Volumes/IA_LAB_DATA/Libreria_Rescate"]:
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

        return {
            "status": "success",
            "stats": {
                "total_files": physical_count,
                "total_chunks": indexed_files,
                "categories": categories
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def semantic_search(query: str, limit: int = 3):
    """Búsqueda holográfica semántica dentro de los chunks de libros."""
    try:
        raw_results = knowledge_bridge.get_raw_results(query, top_k=limit)
        
        sources = list(set([os.path.basename(r['source']) for r in raw_results]))
        
        # Construir respuesta formateada
        formatted_response = "💡 Sabiduría Extraída de tu Bóveda:\n\n"
        for r in raw_results:
            filename = os.path.basename(r['source'])
            clean_chunk = "".join(char for char in r['text'] if char.isprintable() or char in "\n\r\t")
            formatted_response += f"📖 FUENTE: {filename}\n"
            formatted_response += f"└─ {clean_chunk}\n\n"
            
        return {
            "status": "success",
            "response": formatted_response,
            "sources": sources,
            "results": raw_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_endpoint(file_path: str):
    """Procesa, fragmenta e indexa un nuevo archivo en tu base SQLite."""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"El archivo físico no existe en la ruta: {file_path}")
        
        # Importamos y ejecutamos de forma dinámica el ingestor real
        import importlib
        ingest_mod = importlib.import_module("knowledge_engine.diamond_ingestor")
        ingest_mod.ingest_file(file_path)
        
        return {
            "status": "success",
            "message": f"Archivo {os.path.basename(file_path)} indexado en la Bóveda."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =================================================================
# 3. UTILERÍAS DE ARCHIVOS (Con corrección de SyntaxErrors)
# =================================================================

@app.get("/api/files/content")
def get_file_content(path: str):
    """Devuelve el contenido del archivo para previsualización."""
    try:
        if os.path.exists(path):
            from knowledge_engine.diamond_indexer import extract_text
            content = extract_text(path)
            # Devolver solo los primeros 5000 caracteres para el preview
            return {
                "path": path,
                "content": content[:5000],
                "full_length": len(content)
            }
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de lectura: {str(e)}")

@app.get("/api/files/download")
def download_file(path: str):
    """Permite la descarga física de un libro del Lab."""
    from fastapi.responses import FileResponse
    try:
        if os.path.exists(path):
            return FileResponse(path=path, filename=os.path.basename(path))
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en descarga: {str(e)}")


# Entrada de ejecución local con recarga automática
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
