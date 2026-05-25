import os
import json
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Superhuman Core Engine")

# Habilitar CORS para conectar con el frontend en React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPERHUMAN_API_KEY = os.environ.get("SUPERHUMAN_API_KEY", "TU_API_KEY_SECRETA_AQUI")

# Esquemas de Datos Pydantic
class AuditRequest(BaseModel):
    total_spend: float
    total_conversions: int
    unit_value: float
    threshold_percent: float = 75.0  # Límite por defecto (75%)

class CopyRequest(BaseModel):
    category: str
    target_goal: str
    daily_budget: float

# Endpoint 1: Estado del Sistema
@app.get("/api/status")
async def get_status():
    return {
        "status": "Operational",
        "engine": "Superhuman Core Framework",
        "llm_connected": GOOGLE_API_KEY is not None,
        "security_active": SUPERHUMAN_API_KEY != "TU_API_KEY_SECRETA_AQUI"
    }

# Endpoint 2: Auditor de Métricas / Rendimiento Financiero (Semáforo)
@app.post("/api/audit-metrics")
async def audit_metrics(metrics: AuditRequest, x_api_key: str = Header(...)):
    # Validación de seguridad
    if x_api_key != SUPERHUMAN_API_KEY:
        raise HTTPException(status_code=403, detail="Acceso denegado")
    
    # Calcular costo por conversión
    cost_per_conversion = (
        metrics.total_spend / metrics.total_conversions 
        if metrics.total_conversions > 0 
        else metrics.total_spend
    )
    
    limit_threshold = metrics.unit_value * (metrics.threshold_percent / 100.0)
    limit_50 = metrics.unit_value * 0.50
    
    # Determinar estado de semáforo
    if cost_per_conversion >= limit_threshold:
        status = "ROJO"
        action = "PAUSAR"
        alert_message = f"¡ADVERTENCIA! El costo por conversión (${cost_per_conversion:.2f}) superó el límite de seguridad establecido del {metrics.threshold_percent}%. Pausar campaña inmediatamente."
    elif cost_per_conversion >= limit_50:
        status = "AMARILLO"
        action = "OPTIMIZAR"
        alert_message = f"Precaución. La campaña está operando en zona de riesgo. Optimizar creativos o segmentación en las próximas 24 horas."
    else:
        status = "VERDE"
        action = "ESCALAR"
        alert_message = f"Saludable. Rendimiento óptimo. Mantener presupuesto y considerar escalado gradual del 10%."
        
    return {
        "status": status,
        "cost_per_conversion": cost_per_conversion,
        "limit_threshold": limit_threshold,
        "alert_message": alert_message,
        "action": action
    }

# Endpoint 3: Generador de Contenido Persuasivo (Integración LLM)
@app.post("/api/generate-copy")
async def generate_copy(request: CopyRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY no configurado en el servidor."
        )
        
    system_instruction = """Eres un redactor experto en marketing de respuesta directa (copywriting). 
    Tu objetivo es generar una propuesta de anuncio persuasiva, corta y optimizada para conversión en redes sociales.
    
    Debes devolver OBLIGATORIAMENTE un JSON válido (sin bloques de código markdown ```json, sin texto explicativo fuera del JSON) con exactamente estas tres claves:
    {
      "headline": "Un titular llamativo y potente (máximo 10 palabras).",
      "body": "El cuerpo del anuncio (máximo 60 palabras) estructurado con Gancho, Beneficio, y Llamado a la Acción claro.",
      "cta": "El texto del botón de acción más adecuado (ej: 'Comprar', 'Registrarse', 'Más información', 'Enviar mensaje')."
    }
    Responde solo con el JSON crudo en español."""

    prompt = f"""Genera el copy para este negocio:
    - Categoría/Nicho: {request.category}
    - Objetivo del Anuncio: {request.target_goal}
    - Presupuesto Diario: {request.daily_budget}
    
    Devuelve el JSON crudo."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        res_data = response.json()
        
        candidates = res_data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                text_response = parts[0].get("text", "").strip()
                # Verificar validez de JSON antes de retornar
                parsed_json = json.loads(text_response)
                return parsed_json
                
        raise HTTPException(status_code=502, detail="Estructura de respuesta inválida desde la API de Gemini")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error en el motor cognitivo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Correr servidor
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
