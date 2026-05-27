# Checklist de Implementación — Simulador Diamante 💎

## Fase 1: Backend (FastAPI)
- [x] **T1.1**: Crear el modelo Pydantic para el request de simulación en `api/main.py`.
- [x] **T1.2**: Implementar el endpoint `POST /api/simulate`.
- [x] **T1.3**: Implementar el endpoint `POST /api/simulate/reset` para limpiar leads.
- [x] **T1.4**: Verificar funcionamiento con `curl` o Postman.

## Fase 2: Frontend (React)
- [x] **T2.1**: Limpiar `SalesSimulator.tsx` eliminando el iframe y los elementos de Streamlit.
- [x] **T2.2**: Crear el sub-componente `ChatBubble` con estilos Estándar Diamante.
- [x] **T2.3**: Implementar la lógica de envío de mensajes y manejo de estado de carga.
- [x] **T2.4**: Integrar indicadores visuales de Agente y Estado (Badge dinámico).
- [x] **T2.5**: Añadir animaciones de entrada/salida para los mensajes.

## Fase 3: Pulido y Verificación
- [x] **T3.1**: Validar persistencia: recargar la página y ver si los mensajes se mantienen.
- [x] **T3.2**: Probar transiciones de estado (Qualifier -> Closer).
- [x] **T3.3**: Auditoría visual final (márgenes, colores oro/tinta, responsividad).

**Estado Final: COMPLETADO 💎**
