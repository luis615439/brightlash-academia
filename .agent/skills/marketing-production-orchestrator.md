---
name: marketing-production-orchestrator
description: Orquestador de campañas visuales de alta fidelidad para JustLash.
triggers:
  - "crear campaña"
  - "generar variaciones de modelo"
  - "producción híbrida"
  - "escalar assets de marketing"
---

# 🏗️ Protocolo de Orquestación de Producción

## 1. Anclaje de Identidad (Ground Truth)
- **Referencia Obligatoria:** Siempre usar `./assets/examples/laura/diamante.webp`.
- **Anclas Biométricas:** Lunar en pómulo derecho, forma de ojos almendrada.
- **Parámetro Crítico:** `guidance_scale: 9.5` (No negociable para evitar drift).

## 2. Flujo de Subagentes
1. **Generador:** Crea las variantes técnicas (Classic, Hybrid, Volume).
2. **Verificador:** Compara cada output con el 'Diamante'. Si el lunar se desplaza >2px, rechaza y reintenta con mayor peso en el prompt negativo.
3. **Router:** Si la tarea pide video, redirigir al nodo de Video-to-Video (Runway/Luma) vía n8n; no intentar en Vertex estático.

## 3. Control de Calidad Académica
- Las pestañas deben mostrar **aislamiento perfecto**.
- La iluminación debe ser **clínica/studio**, evitando filtros de belleza que suavicen la textura de la piel (necesitamos ver el poro para realismo técnico).

## 🎯 Instrucción de Cierre
"Cada vez que una campaña termine con éxito, el agente debe preguntar si el nuevo asset se convierte en un nuevo 'Diamante' para la base de conocimientos."
