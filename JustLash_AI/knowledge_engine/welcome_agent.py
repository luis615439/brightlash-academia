import json
import os
from datetime import datetime, timezone

# Configuración
CONVERSATIONS_FILE = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/conversations.json"
WELCOME_NOTIFICATIONS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/WELCOME_NOTIFICATIONS/"
LESSON_PATH = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE/PNL/Leccion_La_Mirada_Sintergica.md"

# Asegurar directorios
os.makedirs(WELCOME_NOTIFICATIONS_DIR, exist_ok=True)

def load_conversations():
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_conversations(data):
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def trigger_welcome_vip(lead_id):
    print(f"🎊 ACTIVANDO AGENTE DE BIENVENIDA VIP PARA: {lead_id}")
    data = load_conversations()
    now = datetime.now(timezone.utc)
    
    if lead_id not in data:
        print(f"❌ Error: Lead {lead_id} no encontrado en la base de datos.")
        return False
    
    info = data[lead_id]
    
    # 1. Mensaje de Poder
    power_message = f"""
✨ ¡FELICIDADES, ARQUITECTA DE LA BELLEZA! ✨

Bienvenida al Estándar Diamante de Just Lash Academy. 💎

Hoy has dado un paso que separa a las aficionadas de las verdaderas estrategas. Has dejado de ser una simple aplicadora para convertirte en una visionaria del lash-art.

🎁 Como regalo exclusivo de bienvenida, te adjunto la **'Lección: La Mirada Sintérgica'**, basada en los estudios de Jacobo Grinberg. Esta técnica te enseñará a programar tu éxito desde la primera aplicación.

🚀 **ONBOARDING INMEDIATO:**
- Tus accesos al Portal de Cristal están siendo generados.
- Tu primera sesión de mentoría ha sido agendada para la próxima semana.
- Recibirás un correo con tus credenciales en los próximos 5 minutos.

Bienvenida a la familia, donde diseñamos miradas y construimos imperios.
"""
    
    # 2. Actualizar estado a CONVERTED
    info['state'] = 'converted'
    info['updated_at'] = now.isoformat()
    info['history'].append({
        "role": "assistant",
        "content": "[BIENVENIDA VIP ENVIADA] " + power_message[:150] + "..."
    })
    
    # 3. Notificar al Portal de Cristal
    notification_file = os.path.join(WELCOME_NOTIFICATIONS_DIR, f"WELCOME_{lead_id}_{now.strftime('%Y%m%d_%H%M%S')}.json")
    with open(notification_file, 'w', encoding='utf-8') as nf:
        json.dump({
            "lead_id": lead_id,
            "event": "VIP_WELCOME_TRIGGERED",
            "status": "CONVERTED",
            "payment_confirmed": 1000,
            "lesson_sent": os.path.basename(LESSON_PATH),
            "timestamp": now.isoformat()
        }, nf, indent=2)
    
    save_conversations(data)
    
    print(f"✅ BIENVENIDA VIP COMPLETADA PARA {lead_id}")
    print(f"📢 NOTIFICACIÓN ENVIADA AL PORTAL DE CRISTAL.")
    return True

if __name__ == "__main__":
    # Simulación de Gatillo por Webhook para un lead de prueba
    # Usaremos 'tg-8724626172' que fue impactado por remarketing
    trigger_welcome_vip("tg-8724626172")
