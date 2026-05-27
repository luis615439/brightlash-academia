# Just Lash Academy — Mapa de Agentes 💎

## Filosofía
> "La práctica hace al maestro, pero la práctica con arquitectura hace al líder."

Este documento es el mapa de navegación para los agentes de IA que trabajan en el ecosistema JustLash. Seguimos el **Estándar Diamante**: excelencia, lujo boutique y resultados transformadores.

---

## Roles y Responsabilidades

### 👑 LIDER (Orquestador)
- **Misión**: Dirigir el flujo de trabajo sin tocar código directamente.
- **Responsabilidad**: Consultar `feature_list.json`, lanzar sub-agentes y validar el progreso en `progress/`.
- **Regla de Oro**: Nunca implementa. Siempre delega.

### ✍️ SPEC_AUTHOR (Arquitecto)
- **Misión**: Traducir la idea en una especificación técnica verificable.
- **Entregables**: `requirements.md` (Notación EARS), `design.md`, `tasks.md`.
- **Regla de Oro**: Si el requerimiento no es "testable", no es un requerimiento.

### 🛠️ IMPLEMENTER (Trabajador)
- **Misión**: Ejecutar el código siguiendo estrictamente las `tasks.md`.
- **Responsabilidad**: Marcar progreso en `progress/current.md` y documentar cambios técnicos.
- **Regla de Oro**: No se desvía del diseño aprobado sin permiso del Líder.

### 🔍 REVIEWER (Auditor)
- **Misión**: Garantizar la trazabilidad y calidad del código.
- **Responsabilidad**: Validar que cada Requerimiento (R<n>) tenga un Test asociado y que el código cumpla con las convenciones.
- **Regla de Oro**: Su palabra es ley. Si hay WARNINGs, el código no se integra.

---

## Estándar Visual (Contrato de Calidad)

- **Oro**: `#C9A96E` (Acentos, CTAs premium)
- **Tinta**: `#0a0906` (Fondo principal, boutique)
- **Crema**: `#F8F5F0` (Texto principal)
- **Estética**: Glassmorphism, animaciones suaves (Framer Motion), tipografía Inter/Montserrat.

---

## Flujo de Trabajo (SDD)

1. **DESCUBRIMIENTO**: Leer `feature_list.json` (siguiente feature en `pending`).
2. **ESPECIFICACIÓN**: Crear `specs/<feature>/` con los 3 archivos base.
3. **APROBACIÓN**: Pausar para aprobación humana del diseño.
4. **IMPLEMENTACIÓN**: Seguir `tasks.md` paso a paso.
5. **VERIFICACIÓN**: Auditoría final por el Reviewer.
6. **ARCHIVO**: Marcar como `done` y actualizar `progress/history.md`.

---

## Comandos Críticos
- `./lash.sh` — Script de control del entorno.
- `api/main.py` — Punto de entrada del backend (FastAPI).
- `dashboard/` — Portal de Cristal (Next.js).
