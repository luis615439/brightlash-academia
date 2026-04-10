---
description: Generación de landing pages premium "One Shot" con protocolo de briefing previo
---

# Workflow: Landing "One Shot"

Este workflow se activa cuando el usuario solicita una **"Landing"**, **"landing page"**, o variantes como *"hazme una landing para..."*.

**IMPORTANTE:** Nunca generes código de inmediato. Siempre sigue este protocolo en orden.

---

## Paso 1 — Presentar el Cuestionario de Briefing

Muestra este cuestionario al usuario y espera sus respuestas antes de continuar:

```
🎯 Briefing Just Lash — Sistema One Shot

Antes de generar tu landing, necesito estos datos:

1. **Proyecto** — ¿Cómo se llama la marca?

2. **Producto / Servicio** — ¿Qué vendemos exactamente?
   (Ej. Extensiones de pestañas, cursos online, consultoría)

3. **Audiencia** — ¿A quién va dirigido?
   (Ej. Emprendedoras 25-40, dueños de restaurantes, estudiantes de diseño)

4. **Propuesta de Valor** — ¿Qué nos hace diferentes o premium?
   (Ej. Adhesivos de grado médico, entrega en 24h, garantía de resultados)

5. **Prueba Social** — ¿Algún dato de confianza?
   (Ej. "+500 clientas", "6 años de experiencia", "40 proyectos entregados")

6. **Call to Action Principal** — ¿Qué acción queremos al final?
   (Ej. WhatsApp, Reservar cita, Ver catálogo, Comprar curso)

7. **Estilo Visual** — Elige uno:
   - 🎬 Boutique Cinematográfico (oscuro, dorado, serif elegante)
   - ⚡ Tech Moderno (gradientes azul-violeta, sans-serif, glassmorphism)
   - 🤍 Minimalista Clínico (blanco roto, negro, tipografía limpia, mucho espacio)
```

---

## Paso 2 — Generar Imágenes con `generate_image`

Una vez recibidas las respuestas, usa la herramienta `generate_image` para crear:

1. **Hero image** — Fotografía cinematográfica del producto/servicio en el estilo elegido.
2. **Imagen secundaria** — Ambiente/estudio/contexto de uso del producto.

Copia las imágenes generadas al directorio de trabajo del proyecto.

---

## Paso 3 — Generar el archivo `index.html`

Crea un único archivo `index.html` en el workspace activo con la siguiente estructura obligatoria:

### Estructura de Secciones (en orden):

1. **`<head>`** — Meta tags SEO completos (title, description, og:tags), Google Fonts (Cormorant Garamond + Inter), CSS variables del design system.

2. **Navbar** — Implementar el patrón `isMounted` en JavaScript:
   - El navbar empieza con `opacity: 0; transform: translateY(-8px)`
   - Se aplica la clase `.mounted` vía `requestAnimationFrame` en `DOMContentLoaded`
   - Scroll-aware: agrega clase `.scrolled` con glassmorphism al hacer scroll > 60px
   - **NUNCA** usar `router.refresh()` ni recargas de página

3. **Hero** — Pantalla completa con:
   - Imagen de fondo generada con parallax sutil on-load
   - Overlay gradient para legibilidad
   - `eyebrow` + `h1` con mix serif/sans + subtítulo
   - 2 CTAs: primario (filled gold) + secundario (outline)
   - Indicador de scroll animado

4. **Brand Strip** — Marquee continuo con sellos de autoridad (certificaciones, datos de prueba social)

5. **Problema vs. Solución** — Sección de dos columnas:
   - Izquierda: El "dolor" del cliente antes
   - Derecha: La transformación con el producto/servicio

6. **Grid de Servicios / Productos** — 3 o 4 cards con:
   - Icono decorativo
   - Nombre, descripción, precio o CTA
   - Hover effect: top border gold + background shift
   - Micro-interacción al hover del ícono (glow sutil)

7. **Sección de Autoridad** — Split layout 50/50:
   - Visual: imagen del estudio/producto con badge overlay
   - Contenido: eyebrow + h2 + descripción + grid de 4 stats con contadores animados (IntersectionObserver)

8. **Proceso / Pasos** — 4 pasos enumerados con `step-number` decorativo en serif gigante

9. **Testimonios** — 3 cards con: estrellas, cita en italic serif, avatar inicial + nombre + rol

10. **CTA Final (Cierre)** — Centrado, con:
    - `radial-gradient` glow de fondo
    - `h2` serif grande + subtítulo
    - Botón primario dorado + botón secundario outline
    - Nota de disponibilidad/horario

11. **Footer** — 4 columnas: descripción de marca | servicios | estudio | contacto. Bottom bar con copyright + iconos de redes sociales.

---

## Paso 4 — Reglas Técnicas Innegociables

Aplica siempre estas reglas (PORTAL_CONTRACTS):

- ✅ Usar `isMounted` + `requestAnimationFrame` para el navbar
- ✅ Reveal animations con `IntersectionObserver` (clase `.reveal` → `.visible`)
- ✅ Contadores de stats animados via `requestAnimationFrame`
- ✅ `scroll-behavior: smooth` en `html`
- ✅ Google Fonts: **Cormorant Garamond** (serif) + **Inter** (sans)
- ✅ Scroll anchor offset de 80px para compensar el navbar fijo
- ❌ NUNCA usar `router.refresh()`
- ❌ NUNCA usar Tailwind a menos que el usuario lo pida explícitamente
- ❌ NUNCA dejar imágenes placeholder — siempre generar con `generate_image`

---

## Paso 5 — Previsualización

Después de escribir el archivo, lanza un `browser_subagent` para:
1. Abrir `file:///ruta/al/index.html`
2. Capturar screenshots de Hero, Servicios, Autoridad y CTA Final
3. Reportar cualquier imagen rota o problema visual

---

## Paletas por Estilo

### 🎬 Boutique Cinematográfico
```css
--ink: #0a0906; --gold: #C9A96E; --gold-light: #E2C99A;
--cream: #FDFAF5; --text-muted: #7a7068;
```
Fonts: Cormorant Garamond 300/400 italic + Inter 300/500

### ⚡ Tech Moderno
```css
--bg: #06060f; --accent: #7c3aed; --accent-light: #a78bfa;
--surface: #0f0f23; --text: #e2e8f0; --text-muted: #64748b;
```
Fonts: Space Grotesk 400/600 + Inter 300/500

### 🤍 Minimalista Clínico
```css
--bg: #fafaf9; --ink: #111110; --accent: #18181b;
--muted: #71717a; --border: #e4e4e7;
```
Fonts: Playfair Display 400/700 + Inter 300/400
