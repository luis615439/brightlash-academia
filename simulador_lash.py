"""
Simulador de Ventas — Lash Master CDMX 💎
=========================================
Aplicación Streamlit para que el equipo de ventas practique
cierres siguiendo el contrato de AGENTS.md.

Requisitos:
    pip install streamlit pandas

Ejecución:
    streamlit run simulador_lash.py
"""
import streamlit as st
import pandas as pd
import random
import os
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Lash Master CDMX — Simulador de Ventas",
    page_icon="💎",
    layout="centered",
)

# Estilos Ink & Gold
st.markdown("""
<style>
    .stApp {
        background-color: #0a0906;
        color: #F8F5F0;
    }
    h1, h2, h3 {
        color: #C9A96E !important;
        font-family: 'Playfair Display', serif;
    }
    .stChatMessage {
        background-color: #1a1a14;
        border: 1px solid #C9A96E33;
        border-radius: 12px;
    }
    .stButton > button {
        background-color: #C9A96E;
        color: #0a0906;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #d4b87e;
    }
    .score-box {
        background: linear-gradient(135deg, #C9A96E22, #0a090600);
        border: 1px solid #C9A96E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .score-number {
        font-size: 3em;
        font-weight: bold;
        color: #C9A96E;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATOS DEL SIMULADOR
# ============================================================================
CSV_PATH = os.path.join(os.path.dirname(__file__), "practicas.csv")

PERFILES_1A = [
    {
        "nombre": "María López",
        "edad": 23,
        "tipo": "1A (Principiante)",
        "contexto": "Acaba de terminar la prepa y busca una carrera rápida con buena salida laboral. No tiene experiencia en belleza.",
        "personalidad": "Curiosa pero indecisa. Necesita seguridad."
    },
    {
        "nombre": "Ana Torres",
        "edad": 28,
        "tipo": "1A (Principiante)",
        "contexto": "Trabaja en una oficina pero sueña con independizarse. Le gustan las pestañas pero nunca las ha aplicado.",
        "personalidad": "Motivada pero cautelosa con el dinero."
    },
    {
        "nombre": "Sofía Ramírez",
        "edad": 19,
        "tipo": "1A (Principiante)",
        "contexto": "Influencer pequeña en Instagram. Quiere aprender pestañas para hacérselas ella misma y a sus amigas.",
        "personalidad": "Entusiasta, visual, quiere resultados rápidos."
    },
]

PERFILES_2A = [
    {
        "nombre": "Laura Mendoza",
        "edad": 31,
        "tipo": "2A (Experta)",
        "contexto": "Tiene 3 años de experiencia con clásicas y volumen. Quiere aprender técnicas nuevas como Anime y Wet Look.",
        "personalidad": "Profesional, directa, busca certificaciones."
    },
    {
        "nombre": "Daniela Cruz",
        "edad": 26,
        "tipo": "2A (Experta)",
        "contexto": "Tiene su propio estudio pequeño. Siente que se quedó estancada y necesita técnicas de tendencia para competir.",
        "personalidad": "Ambiciosa pero frustrada con su nivel actual."
    },
    {
        "nombre": "Karla Vega",
        "edad": 34,
        "tipo": "2A (Experta)",
        "contexto": "Instructora en otra academia, pero quiere la certificación de Just Lash por su prestigio. Conoce Ruso y Koda.",
        "personalidad": "Segura de sí misma, exigente, busca el mejor estándar."
    },
]

# ============================================================================
# FUNCIONES DE EVALUACIÓN
# ============================================================================
def evaluar_practica(historial: list, perfil_tipo: str) -> tuple[float, list]:
    """
    Evalúa la conversación de venta según los criterios de AGENTS.md.
    Retorna (puntuación, lista de observaciones).
    """
    puntos = 0.0
    observaciones = []
    texto_completo = " ".join(
        [msg["content"].lower() for msg in historial if msg["role"] == "user"]
    )

    # 1. Metro Balbuena (2.5 pts)
    if "balbuena" in texto_completo:
        puntos += 2.5
        observaciones.append("✅ Mencionó Metro Balbuena")
    else:
        observaciones.append("❌ NO mencionó la ubicación (Metro Balbuena)")

    # 2. Kit de Bienvenida (2.5 pts)
    if "kit" in texto_completo and "bienvenida" in texto_completo:
        puntos += 2.5
        observaciones.append("✅ Ofreció Kit de Bienvenida")
    else:
        observaciones.append("❌ NO ofreció el Kit de Bienvenida")

    # 3. Apartado de $1000 (2.5 pts)
    if "1000" in texto_completo and any(
        w in texto_completo for w in ["aparta", "deposito", "depósito", "apartado"]
    ):
        puntos += 2.5
        observaciones.append("✅ Solicitó apartado de $1,000")
    else:
        observaciones.append("❌ NO solicitó el apartado de $1,000")

    # 4. Segmentación inteligente (2.5 pts)
    if perfil_tipo == "1A (Principiante)":
        if any(
            palabra in texto_completo
            for palabra in ["alumna", "exito", "éxito", "experiencia", "ex alumna"]
        ):
            puntos += 2.5
            observaciones.append("✅ Segmentación 1A: mencionó casos de éxito")
        else:
            observaciones.append(
                "❌ Segmentación 1A: NO mencionó casos de éxito ni ex alumnas"
            )
    else:
        if any(
            palabra in texto_completo
            for palabra in ["anime", "koda", "ruso", "volumen", "mega", "wet"]
        ):
            puntos += 2.5
            observaciones.append("✅ Segmentación 2A: mencionó técnicas de tendencia")
        else:
            observaciones.append(
                "❌ Segmentación 2A: NO mencionó técnicas (Anime/Koda/Ruso)"
            )

    return puntos, observaciones


def guardar_resultado(vendedora: str, perfil: dict, calificacion: float, observaciones: list):
    """Guarda el resultado en el CSV de prácticas."""
    nueva_fila = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Vendedora": vendedora,
        "Perfil_Lead": f"{perfil['nombre']} ({perfil['tipo']})",
        "Calificacion": calificacion,
        "Observaciones": " | ".join(observaciones),
    }

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    else:
        df = pd.DataFrame([nueva_fila])

    df.to_csv(CSV_PATH, index=False)


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================
st.markdown("# 💎 Lash Master CDMX")
st.markdown("### Simulador de Ventas — Estándar Diamante")
st.markdown("---")

# Estado de sesión
if "fase" not in st.session_state:
    st.session_state.fase = "inicio"
    st.session_state.perfil = None
    st.session_state.historial = []

# --- FASE 1: CONFIGURACIÓN ---
if st.session_state.fase == "inicio":
    st.markdown("#### 🎯 Configuración de la Práctica")

    vendedora = st.text_input("Tu nombre:", placeholder="Ej: Laura")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Lead Principiante (1A)", use_container_width=True):
            st.session_state.perfil = random.choice(PERFILES_1A)
            st.session_state.vendedora = vendedora or "Anónimo"
            st.session_state.fase = "practica"
            st.session_state.historial = []
            st.rerun()
    with col2:
        if st.button("🔵 Lead Experta (2A)", use_container_width=True):
            st.session_state.perfil = random.choice(PERFILES_2A)
            st.session_state.vendedora = vendedora or "Anónimo"
            st.session_state.fase = "practica"
            st.session_state.historial = []
            st.rerun()

# --- FASE 2: PRÁCTICA ---
elif st.session_state.fase == "practica":
    perfil = st.session_state.perfil

    # Info del lead
    st.markdown(f"#### 📋 Lead: **{perfil['nombre']}** ({perfil['tipo']})")
    with st.expander("Ver perfil completo", expanded=True):
        st.markdown(f"- **Edad:** {perfil['edad']} años")
        st.markdown(f"- **Contexto:** {perfil['contexto']}")
        st.markdown(f"- **Personalidad:** {perfil['personalidad']}")

    st.markdown("---")
    st.markdown("#### 💬 Escribe tu discurso de venta")
    st.caption("Simula la conversación como si hablaras con esta persona. Escribe todo lo que le dirías.")

    # Mostrar historial
    for msg in st.session_state.historial:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del chat
    if prompt := st.chat_input("Escribe tu mensaje de venta..."):
        st.session_state.historial.append({"role": "user", "content": prompt})

        # Respuesta simulada del lead
        respuestas = [
            f"Hmm interesante, cuéntame más sobre los cursos.",
            f"¿Y dónde queda exactamente?",
            f"¿Qué incluye? ¿Cuánto cuesta?",
            f"Suena bien, pero ¿cómo sé que es diferente a otras academias?",
            f"OK me interesa, ¿cómo le hago?",
        ]
        idx = min(len(st.session_state.historial) // 2, len(respuestas) - 1)
        respuesta_lead = respuestas[idx]
        st.session_state.historial.append(
            {"role": "assistant", "content": f"**{perfil['nombre']}:** {respuesta_lead}"}
        )
        st.rerun()

    # Botón de finalizar
    if len(st.session_state.historial) >= 4:
        st.markdown("---")
        if st.button("🏁 Finalizar y Evaluar", use_container_width=True):
            st.session_state.fase = "resultado"
            st.rerun()

# --- FASE 3: RESULTADOS ---
elif st.session_state.fase == "resultado":
    perfil = st.session_state.perfil
    calificacion, observaciones = evaluar_practica(
        st.session_state.historial, perfil["tipo"]
    )

    # Guardar en CSV
    guardar_resultado(
        st.session_state.vendedora, perfil, calificacion, observaciones
    )

    # Mostrar resultado
    st.markdown("#### 🏆 Resultado de la Evaluación")

    if calificacion >= 10:
        emoji = "💎"
        nivel = "DIAMANTE"
    elif calificacion >= 7.5:
        emoji = "🥇"
        nivel = "ORO"
    elif calificacion >= 5:
        emoji = "🥈"
        nivel = "PLATA"
    else:
        emoji = "🔴"
        nivel = "NECESITA PRÁCTICA"

    st.markdown(
        f"""
        <div class="score-box">
            <div class="score-number">{calificacion}/10</div>
            <div style="font-size: 1.5em;">{emoji} {nivel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📝 Desglose:")
    for obs in observaciones:
        st.markdown(f"- {obs}")

    st.markdown("---")
    if st.button("🔄 Nueva Práctica", use_container_width=True):
        st.session_state.fase = "inicio"
        st.session_state.historial = []
        st.session_state.perfil = None
        st.rerun()
