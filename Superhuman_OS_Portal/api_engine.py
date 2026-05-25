from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "/Volumes/IA_LAB_DAT/INPUT_ZONE/AGENT_CONFIGS"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

class AuditRequest(BaseModel):
    idea: str
    agent_id: str

class AdCopyRequest(BaseModel):
    budgetDaily: float
    salesGoal: str
    productType: str

class CampaignMetrics(BaseModel):
    total_spend: float
    total_leads: int
    ticket_price: float

def get_agent_filename(agent_id: str) -> str:
    file_map = {
        "alex": "alex_supreme_v2.json",
        "merlin": "merlin_integrated.json",
        "branding": "branding_launch.json",
        "kaizen": "kaizen_engine.json",
        "kaizen_v1": "KAIZEN/kaizen_v1_fundamentals.json",
        "kaizen_v2": "KAIZEN/kaizen_v2_growth.json",
        "profile": "execution_profile.json"
    }
    return file_map.get(agent_id)

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    filename = get_agent_filename(agent_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    full_path = os.path.join(CONFIG_PATH, filename)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"File not found at {full_path}")

@app.post("/api/audit")
async def audit_idea(request: AuditRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY not found in environment. Please configure it."
        )

    agent_filename = get_agent_filename(request.agent_id)
    if not agent_filename:
        raise HTTPException(status_code=400, detail="Invalid agent_id")

    agent_path = os.path.join(CONFIG_PATH, agent_filename)
    if not os.path.exists(agent_path):
        raise HTTPException(status_code=404, detail=f"Agent configuration file not found: {agent_filename}")

    with open(agent_path, 'r', encoding='utf-8') as f:
        agent_data = json.load(f)

    profile_path = os.path.join(CONFIG_PATH, "execution_profile.json")
    profile_data = {}
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)

    human_profile = profile_data.get("human_os_profile", profile_data)
    coach = profile_data.get("camali_coach", {})

    agent_name = request.agent_id.upper()
    system_instruction = f"""Eres un auditor de negocios brutal, despiadado e hiper-realista. Operas bajo la mentalidad de: {agent_name}.

Tus directrices de identidad, reglas operativas y conocimiento son:
{json.dumps(agent_data, indent=2, ensure_ascii=False)}

El fundador al que auditas tiene este perfil operativo y reglas diarias no negociables:
{json.dumps(human_profile, indent=2, ensure_ascii=False)}
Coach de ejecución adicional:
{json.dumps(coach, indent=2, ensure_ascii=False)}

Misión obligatoria:
Analiza la propuesta del usuario con extrema honestidad, sin rodeos, detecta los cuellos de botella (bottlenecks), el autoengaño, la falta de foco y los riesgos financieros o de ejecución. Sé sumamente ácido pero constructivo. 
Devuelve tu roast usando formato Markdown (títulos, negritas, listas). Termina siempre con un apartado titulado "### ACCIONES INMEDIATAS (MÁXIMO 3)" con tres consejos accionables y directos para destrabar el negocio hoy mismo."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Esta es mi idea/proyecto a auditar:\n\n{request.idea}"
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": system_instruction
                }
            ]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_data = response.json()
        
        candidates = res_data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                text_response = parts[0].get("text", "")
                return {"audit": text_response}
                
        raise HTTPException(status_code=502, detail="Invalid response structure from Gemini API")
    except requests.exceptions.RequestException as e:
        detail = str(e)
        if response is not None:
            try:
                detail = response.json().get("error", {}).get("message", detail)
            except Exception:
                detail = response.text
        raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}")

@app.post("/api/generate-ad-copy")
async def generate_ad_copy(request: AdCopyRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY not found in environment."
        )
        
    system_instruction = """Eres un redactor experto en marketing de respuesta directa (copywriting), especializado en los frameworks de Alex Hormozi (ofertas irresistibles) y Robert Cialdini (persuasión, autoridad, reciprocidad y escasez).
    
Tu misión es generar una propuesta de anuncio altamente persuasiva y corta para Facebook, optimizada para móvil.
    
Debes devolver OBLIGATORIAMENTE un JSON válido (sin bloques de código markdown ```json, sin texto explicativo fuera del JSON) con exactamente estas tres claves:
{
  "headline": "Un titular llamativo y potente (máximo 10 palabras) que capte la atención.",
  "body": "El cuerpo del anuncio (máximo 60 palabras) estructurado con: Gancho (Hook) inicial, Beneficio principal/Oferta, y un Llamado a la Acción (CTA) directo y claro.",
  "cta": "El texto del botón de acción más adecuado para el objetivo. Elegir únicamente entre: 'Comprar ahora', 'Enviar mensaje', 'Registrarse', 'Más información', 'Cómo llegar'."
}

Adapta el estilo según el tipo de producto/servicio seleccionado. Responde solo con el JSON crudo en español."""

    prompt = f"""Genera el copy para este negocio:
- Tipo de Producto: {request.productType}
- Objetivo de Ventas: {request.salesGoal}
- Presupuesto Diario: {request.budgetDaily}

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
                # Parse JSON to verify correctness
                parsed_json = json.loads(text_response)
                return parsed_json
                
        raise HTTPException(status_code=502, detail="Invalid response structure from Gemini")
    except Exception as e:
        detail = str(e)
        if 'response' in locals() and response is not None:
            try:
                detail = response.json().get("error", {}).get("message", detail)
            except Exception:
                detail = response.text
        raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}")

@app.post("/api/audit-campaign-performance")
async def audit_performance(metrics: CampaignMetrics, x_api_key: str = Header(...)):
    # 1. Validación de seguridad
    api_key_secret = os.environ.get("SUPERHUMAN_API_KEY", "TU_API_KEY_SECRETA_AQUI")
    if x_api_key != api_key_secret:
        raise HTTPException(status_code=403, detail="Acceso denegado")
    
    # 2. Lógica del Auditor
    cpl_actual = metrics.total_spend / metrics.total_leads if metrics.total_leads > 0 else metrics.total_spend
    limit_75 = metrics.ticket_price * 0.75
    
    # Determinación de estado
    status = "VERDE" if cpl_actual < (metrics.ticket_price * 0.5) else ("AMARILLO" if cpl_actual < limit_75 else "ROJO")
    
    # Mensajes del protocolo
    alerts = {
        "ROJO": "¡ADVERTENCIA! El CPL ha superado el límite de seguridad (75%). Pausar campaña inmediatamente para evitar pérdida de margen.",
        "AMARILLO": "Precaución. La campaña está en el límite operativo. Optimizar creativos o segmentación en las próximas 24 horas.",
        "VERDE": "Saludable. Rendimiento óptimo. Mantener presupuesto y considerar un escalado gradual del 10%."
    }
    
    return {
        "status": status,
        "cpl_actual": cpl_actual,
        "limit_75": limit_75,
        "alert_message": alerts[status],
        "action": "PAUSAR" if status == "ROJO" else "MANTENER/ESCALAR"
    }

@app.get("/api/status")
async def get_status():
    return {
        "status": "Superhuman OS Engine Operational", 
        "config_path": CONFIG_PATH,
        "api_key_loaded": GOOGLE_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
