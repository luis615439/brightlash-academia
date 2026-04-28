---
name: justlash-production-master
description: Orquestador Senior para la producción visual y técnica de JustLash Academy.
triggers:
  - "iniciar producción"
  - "campaña nueva"
  - "consistencia de identidad"
  - "escalar marketing"
---

# 🏗️ Protocolo de Producción "Diamante"

## 1. Ground Truth (Verdad Absoluta)
- **Sujeto Principal:** Laura.
- **Referencia Inmutable:** `./assets/examples/laura/diamante.webp`.
- **Anclas Biométricas:** Lunar en el pómulo derecho y ojos almendrados. 
- **Regla de Identidad:** Si el lunar no está presente o la geometría facial cambia, la imagen se marca como **RECHAZADA**.

## 2. Configuración de Motor (Vertex AI + n8n)
- **Guidance Scale:** 9.5 (Fijo para fidelidad máxima).
- **Prompt Upsampling:** OFF (Evitar que la IA "invente" detalles).
- **Iluminación:** Fotografía clínica/studio. Se debe apreciar la textura real de la piel y el aislamiento individual de las pestañas.

## 3. Flujo de Trabajo (Orquestación)
1. **Fase de Generación:** Crear variantes (Clásicas, Híbridas, Volumen).
2. **Fase de Verificación:** Activar subagente para comparar cada output contra el archivo `diamante.webp`.
3. **Fase de Routing:** - Estático -> Vertex AI.
   - Movimiento/Video -> Derivar a nodo Runway/Luma vía n8n (No usar motor estático).

## 4. Control de Calidad Académica
- Las pestañas deben mostrar **aislamiento perfecto** y **retención visual de 6 semanas**.
- Prohibido el uso de filtros "beauty" que borren la técnica de aplicación.

---
"Este protocolo asegura que JustLash mantenga un estándar de calidad Senior en cada pixel generado."
