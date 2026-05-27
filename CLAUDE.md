# JustLash SaaS - Mapa Maestro 💎

Este es el documento central (Ground Truth Arquitectónico) para la academia JustLash.

## 🎯 Objetivo Core
Automatizar y escalar la academia JustLash manteniendo un estándar de calidad **Diamante**, combinando automatización de ventas inteligente y generación de contenido visual hiper-realista.

## 📂 Arquitectura de Agentes
- **Ventas (Qualifier & Closer):** Gestionan el funnel de conversión (HOT/WARM/COLD). Ver `AGENTS.md`.
- **Destilador Maestro:** Alquimia de contenido basada en la Diamond Vault (416 activos). Ver `JustLash_AI/knowledge_engine/`.
- **Producción Visual:** Orquestación Vertex AI + n8n para realismo total.

## 🛠️ Stack Tecnológico
Todo el sistema está gobernado por las reglas definidas en:
- `.cursorrules` (Identidad y Estándar Visual)
- `.agent/rules/stack-tecnico.md` (Integraciones Técnicas)
- `.atl/skill-registry.md` (Contratos de Diseño)
