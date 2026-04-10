# Skill Registry: El Ángel Guardián (Estándar Diamante)

Este documento establece los contratos irrenunciables de arquitectura y diseño para Just Lash Academy. Cualquier modificación al proyecto debe ser validada contra estas reglas.

## Reglas Compactas (Guardian Angel)

### 1. Sistema de Diseño Boutique (Visual)
- **Paleta de Colores**: Solo se permiten variables del sistema:
  - `--gold`: `#C9A96E` (Acentos de lujo)
  - `--ink`: `#0a0906` (Fondo cinemático)
  - `--cream`: `#F8F5F0` (Textos y suavidad)
- **Tipografía**: Títulos en `Cormorant Garamond` (italics para énfasis), Cuerpo en `Inter`.
- **Estética**: Bordes suaves (`--radius-md`), sombras doradas sutiles (`--shadow-gold`) y efectos de cristal (`--glass-border`).

### 2. Contrato de UX y Performance
- **Entrance Gate (Preloader)**: 
  - Salida forzada OBLIGATORIA a los **2 segundos**.
  - No depender exclusivamente del evento `window.load`.
  - Aplicar `display: none` después de la animación para liberar el DOM.
- **Scroll Reveal**: Uso de la clase `.reveal` para animaciones de entrada suaves al hacer scroll.

### 3. Responsividad Diamante (Mobile First)
- **Ancho Crítico**: Elementos de alto impacto (Hero, Logos, Transformaciones) deben ocupar el **90% del ancho** en pantallas móviles (< 768px).
- **Tipografía Escalable**: Uso de `clamp()` para tamaños de fuente que fluyan entre escritorio y móvil sin quiebres.

## Triggers de Validación
- Antes de realizar un `push`, el agente debe verificar:
  - [ ] ¿El preloader mantiene la salida de 2s?
  - [ ] ¿Los nuevos colores siguen la paleta HSL/Hex autorizada?
  - [ ] ¿La imagen de transformación está centrada al 90% en móviles?

---
**Arquitectura Gentlemen: Código que se auto-protege.**
