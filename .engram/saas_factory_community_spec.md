# 🏭 SaaS Factory Community - Especificación Técnica (Fase 1)

**Estado:** V1.0 (Backend Mapped)
**Contexto:** YTOPENROUTER Local Environment
**Framework Base:** $100M Offers (Hormozi)

## 1. Arquitectura de Datos (Backend)
- **FastAPI Core:** `api_routes.py` define los endpoints para la captación de ofertas.
- **Rutas Principales:**
  - `POST /api/v1/saas-factory/capture-offer`: Recibe el nicho, audiencia, problema, solución y precio para su evaluación.

## 2. Scripts de Captación (Token-Aware)
- **Scraper Modular:** `scraper_script.py` simula la extracción de dolores y discusiones de arquitectura desde Reddit y Hacker News.
- **Optimización:** Calcula el peso en tokens antes de procesar para proteger la memoria y evitar sobrecargas en la API.

## 3. Validación y Seguridad
- **Functional Validator:** `validator.py` orquesta la comprobación de integridad.
- **Entorno Seguro:** Validaciones con variables de entorno simuladas (`SAAS_FACTORY_SECRET`) para verificar rutas y flujos de scraping en verde.

## 4. Notas de Ejecución
- Fase 1 completada bajo el protocolo anti-crash (Sprint Modular).
- RAM estable. No se ha invocado ningún componente de frontend/React aún.
