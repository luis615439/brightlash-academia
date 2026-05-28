#!/usr/bin/env python3
"""test_supabase.py – Verificación rápida de conexión y upsert a Supabase.

Uso:
    python3 test_supabase.py

Requiere las variables de entorno:
    SUPABASE_URL, SUPABASE_KEY, TELEGRAM_TOKEN (opcional).

El script intenta upsertar un lead de prueba y captura
`postgrest.exceptions.APIError` para reportar errores de la API.
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv  # ← IMPORTANTE
from supabase import create_client, Client
from postgrest.exceptions import APIError

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL y SUPABASE_KEY deben estar definidas en el entorno.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def test_upsert():
    lead_id = "tg-test-supabase"
    payload = {
        "telefono": lead_id,
        "estado": "Nuevo",
        "origen": "TestScript",
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: supabase.table("leads_justlash").upsert(payload, on_conflict=["telefono"]).execute()
        )
        logging.info("✅ Upsert exitoso: %s", response)
    except APIError as e:
        raise RuntimeError(f"Error de Supabase (APIError): {e.message} (Código: {e.code})")
    except Exception as e:
        raise RuntimeError(f"Error inesperado al conectar con Supabase: {e}")

if __name__ == "__main__":
    asyncio.run(test_upsert())
