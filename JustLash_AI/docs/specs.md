# Spec-Driven Development (SDD) — Guía JustLash 💎

Para garantizar que el software sea tan preciso como una aplicación de pestañas, usamos SDD.

## 1. Notación EARS (Easy Approach to Requirements Syntax)

Cada requerimiento debe seguir uno de estos patrones:

- **Ubiquitous**: The <system name> shall <system response>.
  - *Ej: El sistema deberá cargar los recursos del portal al inicio.*
- **Event-driven**: When <trigger>, the <system name> shall <system response>.
  - *Ej: Cuando el usuario haga clic en un recurso, el simulador deberá cargar su contenido.*
- **State-driven**: While <in state>, the <system name> shall <system response>.
  - *Ej: Mientras el escaneo esté activo, el botón de escaneo deberá estar deshabilitado.*
- **Unwanted Behavior**: If <condition>, then the <system name> shall <system response>.
  - *Ej: Si el archivo no existe, el sistema deberá mostrar un mensaje de error 404.*

## 2. Los 3 Archivos de Especificación

Cada feature en `specs/<feature>/` debe tener:

1. **requirements.md**: Lista numerada (R1, R2, ...) con notación EARS.
2. **design.md**: Diagramas (Mermaid), decisiones de arquitectura, cambios en DB/API.
3. **tasks.md**: Checklist de implementación paso a paso (T1, T2, ...).

## 3. Puerta de Calidad (Quality Gate)

El **Implementador** no puede empezar hasta que el **Líder** (u humano) apruebe los specs. El **Revisor** valida la trazabilidad:
`Requerimiento R1` ↔ `Tarea T1` ↔ `Test de Integración`.

## 4. Estado en Disco

El progreso se registra en `progress/current.md`. Si el agente se reinicia, lee este archivo para saber exactamente dónde quedó.
