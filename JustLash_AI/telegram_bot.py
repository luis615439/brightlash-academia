"""
telegram_bot.py — Integración de JustLash AI con Telegram & Supabase CRM 💎
==========================================================================
Conecta la API local (puerto 8000) de JustLash y gestiona de forma directa
los leads, el historial de interacciones y los estados de conversión en Supabase.
"""

import os
import logging
import asyncio
from typing import Optional
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from supabase import create_client, Client
from postgrest.exceptions import APIError

# Configuración de logs
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")

# Inicializar cliente de Supabase
supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Cliente de Supabase inicializado correctamente.")
    except Exception as e:
        logging.error(f"❌ Error al inicializar cliente de Supabase: {e}")
else:
    logging.warning("⚠️ SUPABASE_URL y/o SUPABASE_KEY no definidos en el entorno. Funcionando sin CRM.")


def determine_new_status(user_message: str, state_after: str) -> Optional[str]:
    """
    Analiza palabras clave y estados del enjambre para determinar
    el estado correspondiente en el CRM (leads_justlash.estado).
    """
    msg_lower = user_message.lower()
    
    # 1. Detección de intenciones por palabras clave
    inscribirse_keywords = [
        "quiero inscribirme", "inscribirme", "inscripcion", "inscripción", 
        "quiero entrar", "quiero el curso", "me interesa inscribirme", "inscribir"
    ]
    pago_keywords = [
        "pago realizado", "dónde deposito", "donde deposito", "dónde pago", 
        "donde pago", "dónde transferir", "transferencia", "deposito", 
        "depósito", "pagar", "pagado", "ya deposité", "ya transferí", 
        "comprobante", "pague", "transfe"
    ]
    
    if any(kw in msg_lower for kw in pago_keywords):
        return "Inscrito"
    if any(kw in msg_lower for kw in inscribirse_keywords):
        return "Interesado"
    
    # 2. Mapeo de transiciones del AgentRouter (n8n Switch replacement)
    if state_after == "converted":
        return "Inscrito"
    elif state_after == "lost":
        return "Rechazado"
    elif state_after == "closing":
        return "Cierre"
    elif state_after in ("qualified", "evaluating"):
        return "Interesado"
        
    return None


async def ensure_lead_exists(lead_id: str) -> None:
    """Verifica si el lead existe en leads_justlash. Si no, lo crea como 'Nuevo' usando upsert."""
    if not supabase_client:
        return

    try:
        loop = asyncio.get_event_loop()
        # Upsert evita duplicados basados en la columna 'telefono'
        payload = {
            "telefono": lead_id,
            "estado": "Nuevo",
            "origen": "Telegram",
        }
        # Validación interna del JSON antes de enviarlo
        required_keys = {"telefono", "estado", "origen"}
        if not required_keys.issubset(payload.keys()):
            raise ValueError(f"Payload incompleto para lead {lead_id}: faltan {required_keys - set(payload.keys())}")
        await loop.run_in_executor(
            None,
            lambda: supabase_client.table("leads_justlash").upsert(payload, on_conflict=["telefono"]).execute()
        )
        logging.info(f"🆕/🔄 Lead {lead_id} upsertado como 'Nuevo' en Supabase.")
    except APIError as e:
        raise RuntimeError(f"Error de Supabase al upsertar lead {lead_id}: {e.message} (Código: {e.code})")
    except Exception as e:
        logging.error(f"⚠️ Error al upsertar lead {lead_id} en Supabase: {e}")


async def save_interaction(lead_id: str, user_message: str, ia_response: str) -> None:
    """Registra la interacción de la alumna y la respuesta del modelo en Supabase, añadiendo timestamp si la columna existe."""
    if not supabase_client:
        return

    try:
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        payload = {
            "telefono": lead_id,
            "mensaje_usuario": user_message,
            "respuesta_ia": ia_response,
            "created_at": timestamp,  # columna de tipo timestamp en Supabase
        }
        # Validación interna del JSON
        required_keys = {"telefono", "mensaje_usuario", "respuesta_ia", "created_at"}
        if not required_keys.issubset(payload.keys()):
            raise ValueError(f"Payload incompleto para interacción {lead_id}: faltan {required_keys - set(payload.keys())}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: supabase_client.table("interacciones_justlash").insert(payload).execute()
        )
        logging.info(f"💾 Interacción guardada en Supabase para lead: {lead_id} (timestamp {timestamp})")
    except APIError as e:
        raise RuntimeError(f"Error de Supabase al guardar interacción para lead {lead_id}: {e.message} (Código: {e.code})")
    except Exception as e:
        logging.error(f"⚠️ Error al guardar interacción en Supabase para {lead_id}: {e}")


async def sync_lead_status(lead_id: str, user_message: str, state_after: str) -> None:
    """Actualiza dinámicamente el estado del Lead en Supabase según el análisis."""
    if not supabase_client:
        return
    
    new_status = determine_new_status(user_message, state_after)
    if not new_status:
        return
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: supabase_client.table("leads_justlash").update({
                "estado": new_status
            }).eq("telefono", lead_id).execute()
        )
        logging.info(f"🔄 Estado de lead {lead_id} actualizado a '{new_status}' en Supabase.")
    except Exception as e:
        logging.error(f"⚠️ Error al actualizar estado del lead {lead_id} en Supabase: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa los mensajes entrantes de Telegram llamando a la API local de FastAPI."""
    if not update.message or not update.message.text:
        return

    user = update.effective_user
    lead_id = f"tg-{user.id}"
    user_message = update.message.text

    logging.info(f"📥 Mensaje de {user.first_name} ({lead_id}): {user_message}")

    # Mostrar estado "escribiendo..." en la interfaz de Telegram
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # 1. Asegurar la existencia del Lead en Supabase de forma asíncrona
    await ensure_lead_exists(lead_id)

    # 2. Consultar la API local de FastAPI en el puerto 8000
    response_message = "Lo siento, nena. Estoy teniendo un problema técnico para conectarme con mis compañeras. Por favor, intenta de nuevo en unos minutos. 💎"
    state_after = "qualifying"
    
    try:
        async with httpx.AsyncClient() as client:
            api_response = await client.post(
                API_URL,
                json={"lead_id": lead_id, "message": user_message},
                timeout=30.0
            )
            api_response.raise_for_status()
            result = api_response.json()
            
            # Extraer respuesta y estado
            response_message = result.get("response", response_message)
            state_after = result.get("state_after", state_after)
            
            logging.info(f"🤖 API respondió ({result.get('metadata', {}).get('agent_name', 'Desconocido')}): {response_message[:50]}...")

    except httpx.HTTPStatusError as e:
        logging.error(f"❌ Error HTTP al consultar API de JustLash: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logging.error(f"❌ Error de conexión al consultar API de JustLash: {e}")
    except Exception as e:
        logging.error(f"❌ Error general en comunicación con API: {e}")

    # 3. Enviar la respuesta al usuario en Telegram (incluso si falló la API, responde el fallback)
    try:
        await update.message.reply_text(response_message)
    except Exception as e:
        logging.error(f"❌ Error al enviar mensaje de vuelta en Telegram: {e}")

    # 4. Guardar interacción y actualizar estado en Supabase de forma diferida (sin retrasar la respuesta)
    asyncio.create_task(save_interaction(lead_id, user_message, response_message))
    asyncio.create_task(sync_lead_status(lead_id, user_message, state_after))


if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("❌ ERROR: No se encontró TELEGRAM_TOKEN en el archivo .env")
        exit(1)

    print(f"🚀 Iniciando JustLash Telegram Bot (Supabase CRM Activo)")
    print(f"🔗 Conectado a la API en: {API_URL}")
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(text_handler)
    
    application.run_polling()
