import json
import os
from datetime import datetime, timedelta, timezone
from alchemy_engine import distill_and_convert
import sqlite3

# Configuración
CONVERSATIONS_FILE = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/conversations.json"
ALERTS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/REMARKETING_ALERTS/"
RESOURCES_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/RECURSOS_VENTAS/"
DAYS_INACTIVE = 7

# Asegurar directorios
os.makedirs(ALERTS_DIR, exist_ok=True)

def load_conversations():
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_conversations(data):
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_latest_script():
    # Busca el guion más reciente en Recursos de Ventas
    scripts = [f for f in os.listdir(RESOURCES_DIR) if f.startswith("GUION_") and f.endswith(".md")]
    if not scripts:
        return None
    # Por simplicidad, tomamos el último (podría ser por fecha de modificación)
    with open(os.path.join(RESOURCES_DIR, scripts[0]), 'r', encoding='utf-8') as f:
        return f.read()

def run_remarketing_mission():
    print("🚀 INICIANDO MISIÓN DE REMARKETING: AGENTE DIAMANTE")
    data = load_conversations()
    now = datetime.now(timezone.utc)
    target_date = now - timedelta(days=DAYS_INACTIVE)
    
    leads_impacted = 0
    script_content = get_latest_script()
    
    if not script_content:
        print("❌ Error: No se encontró un guion de venta aprobado en Recursos de Ventas.")
        return

    for lead_id, info in data.items():
        updated_at = datetime.fromisoformat(info['updated_at'])
        
        # Filtro: Inactivo > 7 días Y no es estado terminal (converted/dead)
        if updated_at < target_date and info['state'] not in ['converted', 'dead']:
            print(f"🎯 IMPACTANDO LEAD: {lead_id} (Inactivo desde {updated_at.date()})")
            
            # Simular impacto (en un sistema real aquí se enviaría el mensaje por Telegram/WhatsApp)
            # Personalizar el guion (ejemplo simple)
            personalized_script = script_content.replace("[nombre]", lead_id)
            
            # Actualizar estado
            info['state'] = 'remarketing'
            info['remarketing_attempt'] = info.get('remarketing_attempt', 0) + 1
            info['updated_at'] = now.isoformat()
            
            # Registrar en historial (Simulado)
            info['history'].append({
                "role": "assistant",
                "content": "[REMARKETING ENVIADO] " + personalized_script[:200] + "..."
            })
            
            leads_impacted += 1
            
            # Reportar al Portal de Cristal (Notificación)
            alert_file = os.path.join(ALERTS_DIR, f"ALERT_{lead_id}_{now.strftime('%Y%m%d_%H%M%S')}.json")
            with open(alert_file, 'w', encoding='utf-8') as af:
                json.dump({
                    "lead_id": lead_id,
                    "action": "REMARKETING_SENT",
                    "timestamp": now.isoformat(),
                    "script_used": "Arquitectos de la Belleza"
                }, af, indent=2)

    save_conversations(data)
    print(f"✅ MISIÓN FINALIZADA. Leads impactados: {leads_impacted}")
    
    # Aquí se guardaría en Engram (Simulado vía log para el usuario)
    print(f"🧠 MEMORIA: Guardando métricas en Engram (Tasa de impacto: {leads_impacted} leads)")

if __name__ == "__main__":
    run_remarketing_mission()
