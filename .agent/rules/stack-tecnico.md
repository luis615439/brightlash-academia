---
name: stack-tecnico
description: Reglas de integración para el Stack de Automatización JustLash.
---

# ⚙️ Stack Técnico "JustLash SaaS"

## 1. n8n (El Cerebro de Automatización)
- **Rol:** Orquestador principal de flujos (Webhooks, enrutamiento condicional).
- **Regla:** Todos los endpoints deben estar documentados y manejar respuestas de error limpias. 
- **Enrutamiento:** Los leads calificados por el Qualifier van al CRM. Los assets de video se derivan a nodos especializados (Runway/Luma).

## 2. Google Vertex AI (El Motor de Imagen)
- **Rol:** Generación de assets estáticos hiper-realistas.
- **Configuración Forzada:**
  - `guidance_scale`: 9.5
  - `prompt_upsampling`: OFF
- **Regla:** Vertex AI solo procesa imágenes estáticas ancladas a una referencia cruzada (ej. `diamante.webp`). No se le pide video.

## 3. Supabase (El Hub de Datos)
- **Rol:** Base de datos relacional y almacenamiento (storage) de assets/rutas de imagen y base de CRM.
- **Regla:** Cada lead HOT calificado debe enviarse como payload JSON a Supabase. Las rutas de las imágenes generadas se guardan referenciando al Lead/Campaña.
