# Prompt Maestro Engine Bridge: Estética Just Lash

Utiliza este prompt en la herramienta `engine-bridge` (o directamente en modelos como Stable Diffusion XL) para procesar imágenes y que se integren orgánicamente con la landing page.

---

## ✦ El Prompt Maestro (Ambient Diffusion)

> **Prompt:** 
> "Professional high-end beauty photography of [SUBJECT], captured in a cinematic obsidian-toned studio. The atmosphere is defined by deep ink blacks (#0a0906) and dramatic chiaroscuro lighting. Soft golden hour glow (#C9A96E) highlights the contours, reflecting off polished glass and obsidian surfaces. Luxurious velvet textures and sharp high-fidelity details. Ultra-realistic skin textures, sharp focus on eyes and lashes. Color grading inspired by 'Estándar Diamante': golden accents on deep charcoal backgrounds. 8k resolution, shot on Hasselblad, hyper-detailed, elegant, boutique aesthetic."

---

## 🛠 Guía de Aplicación Técnica

| Parámetro | Valor Recomendado | Razón |
| :--- | :--- | :--- |
| `model_type` | `background_diffusion` | Para generar el entorno premium alrededor del sujeto. |
| `prompt_ambient` | (Usar el prompt de arriba) | Proporciona el contexto visual "Ink & Gold". |
| `negative_prompt` | "Blurry, grainy, low resolution, amateur, bright colors, neon, plastic, overexposed, messy background." | Mantiene la limpieza y el lujo visual. |

## 💡 Consejos Pro

1. **Restauración Previa**: Si la foto original es de baja calidad o tiene ruido, corre primero un proceso de `restoration-gfpgan` antes de aplicar la ambientación.
2. **Iluminación**: El prompt menciona "chiaroscuro". Esto es vital para que las sombras se fundan con el fondo `#0a0906` de la landing, creando ese efecto de profundidad infinito.
3. **Consistencia**: Al usar el código de color hexadecimal `#C9A96E` directamente en el prompt, obligas a la IA a buscar esa tonalidad exacta de oro.
