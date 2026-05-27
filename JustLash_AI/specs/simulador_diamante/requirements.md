# Requerimientos — Simulador Diamante 💎

## R1: Interfaz de Chat Nativa
- **Ubiquitous**: El sistema deberá presentar una interfaz de chat estilo "WhatsApp Premium" dentro del dashboard.
- **State-driven**: Mientras el sistema esté procesando una respuesta, el input del usuario deberá estar deshabilitado y mostrar un indicador de "Agente escribiendo...".

## R2: Integración con AgentRouter
- **Event-driven**: Cuando el usuario envíe un mensaje, el sistema deberá realizar una petición POST al endpoint `/api/simulate` del backend.
- **Ubiquitous**: El sistema deberá persistir el historial de conversación en `conversations.json` a través del `AgentRouter` existente.

## R3: Visualización de Metadatos del Agente
- **Ubiquitous**: El sistema deberá mostrar qué agente está respondiendo (Qualifier, Closer o Remarketing) y su estado actual (Qualifying, Closing, etc.).
- **Event-driven**: Cuando se detecte una transición de estado (ej: de Qualifying a Qualified), el sistema deberá mostrar una notificación visual de "Transición de Agente".

## R4: Control de Simulación
- **Event-driven**: Cuando el usuario haga clic en "Reiniciar Simulación", el sistema deberá llamar al método `reset_lead` del backend y limpiar la pantalla de chat.
- **Ubiquitous**: El sistema deberá permitir seleccionar un ID de lead existente o generar uno nuevo para iniciar una sesión.

## R5: Manejo de Errores
- **Unwanted Behavior**: Si la API de OpenRouter falla o no hay conexión, el sistema deberá mostrar un mensaje de error elegante y permitir reintentar el envío.
