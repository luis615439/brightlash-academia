---
name: justlash-image-expert
description: Skill experta para la App de JustLash (Vertex AI + n8n).
triggers:
  - "optimizar app de imagen"
  - "error en generacion"
  - "nuevos recursos de imagen"
  - "ajustar prompt de pestañas"
  - "Laura"
  - "Nuevos Recursos"
---

# 👁️ JustLash Image Expert Node

## Contexto Técnico
- **Infraestructura:** Google Vertex AI (reemplazando a Fal.ai).
- **Orquestación:** Webhooks en n8n.
- **Base de Datos:** Supabase para el almacenamiento de rutas de imagen.

## Protocolo de Optimización (Arquitectura Gentleman)
1. **Validación de Prompt:** Antes de enviar a Vertex AI, el agente debe enriquecer el prompt asegurando palabras clave: "macro shot", "8k resolution", "flawless lash alignment", "clinical lighting".
2. **Control de Calidad:** Si la imagen presenta artefactos en el iris o pestañas duplicadas, el agente debe sugerir un ajuste en el parámetro de 'guidance_scale'.
3. **Inyección de Recursos:** [ESPACIO RESERVADO PARA LOS NUEVOS DATOS QUE PROPORCIONARÁS MÁS TARDE].

### 🆔 Protocolo de Fidelidad de Identidad (Diamante)
1.  **Referencia Cruzada Obligatoria:** Si la instrucción incluye "Laura" u otro sujeto con assets, el agente DEBE autoinvocar un subagente para realizar una triangulación visual entre el prompt y los archivos de `/assets/examples`.
2.  **Anclaje de Rasgos:** Identifica y fija los rasgos inmutables del sujeto en el prompt (ej. "lunar en el pómulo derecho", "forma de ojos almendrada"). No delegues esto a la suerte del modelo.
3.  **Ajuste Dinámico de Guidance:** Si el usuario reporta que el parecido bajó, sugiere aumentar el `guidance_scale` a un rango de 9.0-10.0 para forzar al modelo a pegarse más a la descripción escrita que a su "creatividad".

## Instrucción de Autoinvocación
"Cada vez que el usuario mencione 'Laura' o 'Nuevos Recursos', consulta la carpeta /assets/examples de esta skill para mantener la consistencia visual."
