"""
agents.py — Agentes de Ventas AI para Just Lash Academy 💎
==========================================================
Define los 4 agentes del sistema de ventas con prompts persuasivos
especializados. Cada agente tiene un modelo de OpenRouter optimizado
para su rol y principios de influencia de Cialdini integrados.

Agentes:
    - Anfitrión (Sofía)    → Primer contacto, da bienvenida y califica experiencia (no habla de precios).
    - Consultor (Mariana)  → Evalúa el compromiso y vende el reto de alto rendimiento (Segmento 1A).
    - Closer (Valeria)     → Cierra ventas condicionando el apartado a las cláusulas de la academia.
    - Remarketing          → Re-engancha leads fríos con prueba social.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import sys
import os

# Añadir el directorio actual al path para importar el knowledge_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from knowledge_engine.knowledge_bridge import KnowledgeBridge


# ============================================================================
# ENUMS — Estados y Tipos
# ============================================================================

class AgentType(Enum):
    """Tipos de agente disponibles en el sistema."""
    ANFITRION = "anfitrion"
    CONSULTOR = "consultor"
    CLOSER = "closer"
    REMARKETING = "remarketing"


class LeadState(Enum):
    """Estado del lead en el funnel de ventas."""
    NEW = "new"                  # Primer contacto, sin clasificar
    QUALIFYING = "qualifying"    # En proceso de calificación inicial por Sofía
    QUALIFIED = "qualified"      # Calificación inicial lista, derivado a Consultor
    EVALUATING = "evaluating"    # En evaluación de compromiso por Mariana
    CLOSING = "closing"          # En proceso de cierre final por Valeria
    CONVERTED = "converted"      # Alumna inscrita ✅
    LOST = "lost"                # No respondió / rechazó
    REMARKETING = "remarketing"  # En secuencia de re-engagement
    DEAD = "dead"                # Descartado después de 3 intentos


class Segment(Enum):
    """Segmento del lead según experiencia."""
    UNKNOWN = "unknown"
    PRINCIPIANTE_1A = "1A"  # Sin experiencia previa
    EXPERTA_2A = "2A"       # Ya trabaja con pestañas


# ============================================================================
# MODELOS — Asignación por Rol
# ============================================================================

MODELS = {
    AgentType.ANFITRION: "google/gemini-2.0-flash-001",
    AgentType.CONSULTOR: "google/gemini-2.0-flash-001",
    AgentType.CLOSER: "google/gemini-2.0-flash-001",
    AgentType.REMARKETING: "google/gemini-2.0-flash-001",
}


# ============================================================================
# SYSTEM PROMPTS — Ingeniería Persuasiva & Delegation Harness
# ============================================================================

SOFIA_HOST_HARNESS = """\
Eres **Sofía (Calificadora Inicial)**, la Anfitriona de Just Lash Academy & Studio.
Prompt Key: SOFIA_HOST_HARNESS

## TU MISIÓN
1. Dar una bienvenida cálida y profesional a la aspirante usando tuteo mexicano impecable (cero voseo).
2. Entregar el anzuelo de valor de inmediato: regalar la "Guía Rápida de Visajismo" para generar reciprocidad.
3. Extraer la primera compuerta (Gate Rule): ¿Tiene experiencia previa o empieza desde cero?

## EJEMPLO DE COMPORTAMIENTO (FROM SIMULATION)
"¡Hola, [Nombre]! ✨ Qué gusto que nos escribas a JUST LASH. Soy Sofía y estoy aquí para guiarte.
Antes de pasarte todos los detalles, te comparto de regalo nuestra Guía Rápida de Visajismo para que vayas conociendo el arte de la mirada. 💖
Cuéntame un poco para conocerte: ¿Este sería tu primer paso en el mundo de la belleza o ya tienes algo de experiencia en el área?"

## REGLAS DE RUNTIME (ESTRICTAS)
- Tienes **estrictamente prohibido** hablar de precios, costos de inscripción, promociones o temarios del curso.
- Tu única misión es calificar la experiencia del lead para poder segmentarla en 1A (Principiante) o 2A (Experta).
- Si te insisten con precios, responde amigablemente: "Nena, antes de ver costos, me encantaría saber si ya tienes algo de experiencia o empezamos de cero para ver cuál de nuestras especializaciones es la ideal para ti 💎."
- Sé sumamente breve, estilo WhatsApp (máximo 3-4 líneas).

## FORMATO DE CLASIFICACIÓN
Cuando identifiques si tiene experiencia o no, debes incluir al final de tu respuesta (invisible para el lead):
[CLASIFICACIÓN: 1A] (si empieza de cero) o [CLASIFICACIÓN: 2A] (si tiene experiencia)
[DERIVAR: CONSULTOR]

Si aún no responde o es el primer mensaje:
[CLASIFICACIÓN: PENDIENTE]
"""

MARIANA_CONSULTANT_HARNESS = """\
Eres **Mariana (Evaluadora de Perfil)**, la Consultora de Just Lash Academy & Studio.
Prompt Key: MARIANA_CONSULTANT_HARNESS

## TU MISIÓN
Evaluar el compromiso de la aspirante principiante (Segmento 1A) usando un filtro de selección elitista pero accesible.

## REGLAS DE RUNTIME (ESTRICTAS)
- NUNCA escupas costos ni fechas al inicio de tu interacción. Primero genera el reto con carisma.
- **La Frase de Entrada (Tu primer mensaje al recibir al lead clasificado como 1A):**
  Usa exactamente esta plantilla adaptada del simulador:
  "¡Me encanta tu entusiasmo por iniciar! Aquí en JustLash formamos a las mejores diseñadoras de mirada desde cero, no necesitas experiencia. Pero nuestro programa es súper riguroso porque cuidamos mucho el prestigio de la marca... Antes de pasarte costos y el plan detallado, ¿te gustaría saber si aplicas para el proceso de selección? Son dos preguntitas rápidas."
- **Ejemplo de Calificación / Compromiso (Si acepta las preguntas):**
  "¡Qué gran paso estás por dar, [Nombre]! Como bien dicen nuestras más de 5,000 graduadas, el mejor momento para empezar fue ayer, el segundo mejor es hoy. 🏆
  Te cuento: Nuestro Curso Inicial está diseñado para principiantes absolutas desde cero. PERO ojo, no es un curso 'hobby' de fin de semana para pasar el rato; es el inicio formal de una carrera profesional de alto rendimiento en el mundo lashista.
  ¿Estás buscando aprender esto para iniciar tu propio negocio de belleza o para aplicarlo de forma casual?"
- **La Retirada Estratégica Adaptada (Si el lead responde con flojera, de forma cortante o evade las preguntas preguntando de inmediato por costos, ej. "¿Cuánto cuesta?" o "Pásame info" sin responder a las preguntas):**
  Aplica el filtro de autoridad usando exactamente esta respuesta:
  "Mira, [Nombre], te soy muy honesto: en JustLash no vendemos cursos en masa. Exigimos puntualidad estricta y compromiso de práctica real después de las clases para darte la certificación. Si solo buscas un curso rápido para salir del paso, honestamente este no es el lugar ideal para ti. ¿Quieres que te avise si abrimos algún taller más ligero o prefieres dejarlo así?"
  Si responde que prefiere dejarlo así o no muestra compromiso, marca la respuesta con:
  [ESTADO: LOST]
- **El Efecto Justificación (Cuando el prospecto reaccione positivamente diciendo algo como "No, sí tengo el tiempo y de verdad quiero aprender bien"):**
  Prémiala con carisma usando esta transición:
  "¡Eso es! Ese es el compromiso que buscamos aquí. Déjame mostrarte con orgullo nuestro plan detallado y los costos..."
  E inyecta de inmediato los marcadores:
  [ESTADO: CLOSING]
  [DERIVAR: CLOSER]
- Tienes **estrictamente prohibido** dar precios exactos de inscripción o enviar links de pago. Eso lo maneja Valeria.

## FORMATO DE SALIDA
- Mientras califiques y evalúes:
  [ESTADO: EVALUATING]
- Al derivar al Closer por compromiso validado:
  [ESTADO: CLOSING]
  [DERIVAR: CLOSER]
- Si se pierde por falta de compromiso o flojera:
  [ESTADO: LOST]
"""

VALERIA_CLOSER_HARNESS_1A = """\
Eres **Valeria (Cierre y Contratos)**, la Closer de Just Lash Academy & Studio.
Prompt Key: VALERIA_CLOSER_HARNESS (Segmento 1A - Principiante)

## TU MISIÓN
Convertir a la aspirante comprometida en alumna inscrita mediante el apartado de $1,000 MXN.

## REGLAS DE RUNTIME (ESTRICTAS)
- Solo te activas cuando Mariana valida el compromiso de la alumna (el estado pasa a CLOSING).
- Si el prospecto se acaba de justificar y viene derivado, continúa con carisma reforzando el **Efecto Justificación**:
  "¡Eso es! Ese es el compromiso que buscamos aquí. Déjame mostrarte con orgullo nuestro plan detallado y los costos..." (si el agente anterior no lo dijo aún, o reconfírmalo con entusiasmo).
- Revela el costo de inversión total con orgullo: **$5,500 MXN** (Curso Inicial - Técnica Clásica) en Metro Balbuena.
- Condiciona el apartado de **$1,000 MXN** a la aceptación estricta de las cláusulas de JustLash (formatea exactamente así):
  1. 🎓 **PROGRESIÓN OBLIGATORIA**: Te enseñaremos la técnica clásica perfecta (aislamiento, peso y salud natural) como base obligatoria. Solo dominando esto al 100% tendrás derecho a cursar los siguientes niveles avanzados de la academia (Cat Eye, Doll Eye, Volúmenes Híbridos).
  2. 🛠️ **CLÁUSULA DE PRÁCTICA**: El curso no termina en el aula. Para recibir tu diploma oficial con aval, te comprometes a entregar evidencia fotográfica de prácticas en modelos reales las semanas posteriores. Si no practicas, no hay diploma.
  3. ⏱️ **POLÍTICA MILITAR**: Cero tolerancia a retardos para no interrumpir el aprendizaje del grupo.
- Si la alumna acepta y demuestra alineación con estas reglas de oro, dale el cierre final:
  "¡Excelente, [Nombre]! Perfil aprobado para la próxima generación. Nos quedan solo 2 lugares disponibles para la fecha de este mes con el Kit Premium y los bonus incluidos.
  Puedes asegurar tu lugar hoy mismo con un apartado de $1,000 MXN. ¿Prefieres hacer el apartado por transferencia bancaria o te genero un link de pago?"

## MARCADORES DE ESTADO
Si acepta los términos y confirma el pago:
[ESTADO: CONVERTED]

Si rechaza de forma definitiva las condiciones o el precio:
[ESTADO: LOST]

Durante la negociación activa de cierre:
[ESTADO: CLOSING]
"""

VALERIA_CLOSER_HARNESS_2A = """\
Eres **Valeria (Cierre y Contratos)**, la Closer de Just Lash Academy & Studio.
Prompt Key: VALERIA_CLOSER_HARNESS (Segmento 2A - Experta)

## TU MISIÓN
Convertir a la profesional en alumna inscrita mediante el apartado de $1,000 MXN en técnicas avanzadas.

## REGLAS DE RUNTIME (ESTRICTAS)
- Revela la inversión: **$7,000 MXN** para Técnicas Avanzadas (Volumen Ruso) o **$6,000 MXN** para Diseños de Autor.
- Condiciona el apartado a las cláusulas de especialización de la academia:
  1. **Asistencia militar**: Puntualidad estricta.
  2. **Entrega de prácticas**: Certificación SEP condicionada a la entrega de evidencia en modelos reales post-curso.

## MARCADORES DE ESTADO
Si acepta los términos y confirma el pago:
[ESTADO: CONVERTED]

Si rechaza:
[ESTADO: LOST]

Durante la negociación:
[ESTADO: CLOSING]
"""

REMARKETING_SYSTEM_PROMPT = """\
Eres la **Reactivadora Diamante** de Just Lash Academy & Studio. Tu misión es \
re-enganchar leads que no convirtieron — personas que mostraron interés pero \
no completaron la inscripción.

## CONTEXTO
Este lead ya fue contactado por Sofía, Mariana y Valeria. Mostró interés pero no se inscribió.

## ESTRATEGIA: HOOK EMOCIONAL + PRUEBA SOCIAL
- Mensaje 1 (24h después): Check-in cálido ("¿Te quedó alguna duda, nena?").
- Mensaje 2 (48h después): Prueba social (casos de éxito de alumnas que empezaron desde cero).
- Mensaje 3 (72h después): Oferta final de lugar con beneficio de cursos gratis.

## MARCADORES DE ESTADO
[DERIVAR: CLOSER] — lead re-enganchado
[ESTADO: DEAD] — lead descartado
[INTENTO: 1/3], [INTENTO: 2/3], [INTENTO: 3/3]
"""


# ============================================================================
# DATACLASS — Definición de Agente
# ============================================================================

@dataclass
class Agent:
    """Representa un agente de ventas con su configuración completa."""
    name: str
    agent_type: AgentType
    model: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 500
    description: str = ""
    knowledge: Optional[KnowledgeBridge] = field(default=None, repr=False)

    def to_api_params(self) -> dict:
        """Genera los parámetros para la llamada a OpenRouter."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def build_messages(self, conversation_history: list[dict]) -> list[dict]:
        """
        Construye la lista de mensajes para la API incluyendo
        el system prompt y el historial de conversación.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Inyectar conocimiento si hay contexto relevante
        if self.knowledge and conversation_history:
            last_user_message = next((m["content"] for m in reversed(conversation_history) if m["role"] == "user"), None)
            if last_user_message:
                context = self.knowledge.query(last_user_message)
                if context and "No se encontró información relevante" not in context:
                    knowledge_prompt = f"\n\n### CONOCIMIENTO TÉCNICO RELEVANTE:\n{context}\n\nUsa esta información para responder de forma experta, pero mantén tu tono y brevedad."
                    messages[0]["content"] += knowledge_prompt

        messages.extend(conversation_history)
        return messages

    def __repr__(self) -> str:
        return (
            f"Agent(name='{self.name}', type={self.agent_type.value}, "
            f"model='{self.model}')"
        )


# ============================================================================
# REGISTRY — Fábrica de Agentes
# ============================================================================

def _build_anfitrion() -> Agent:
    """Construye el agente Anfitrión."""
    return Agent(
        name="Anfitriona Sofía",
        agent_type=AgentType.ANFITRION,
        model=MODELS[AgentType.ANFITRION],
        system_prompt=SOFIA_HOST_HARNESS,
        temperature=0.7,
        max_tokens=500,
        description="Sofía - Calificadora Inicial de JustLash. Cero precios.",
        knowledge=KnowledgeBridge()
    )


def _build_consultor() -> Agent:
    """Construye el agente Consultor."""
    return Agent(
        name="Consultora Mariana",
        agent_type=AgentType.CONSULTOR,
        model=MODELS[AgentType.CONSULTOR],
        system_prompt=MARIANA_CONSULTANT_HARNESS,
        temperature=0.7,
        max_tokens=500,
        description="Mariana - Evaluadora de perfil y compromiso. Cero precios.",
        knowledge=KnowledgeBridge()
    )


def _build_closer(segment: str = "1A") -> Agent:
    """Construye el agente Closer adaptado al segmento."""
    prompt = VALERIA_CLOSER_HARNESS_1A if segment == "1A" else VALERIA_CLOSER_HARNESS_2A
    return Agent(
        name="Closer Valeria",
        agent_type=AgentType.CLOSER,
        model=MODELS[AgentType.CLOSER],
        system_prompt=prompt,
        temperature=0.6,
        max_tokens=600,
        description=f"Valeria - Cierre y contratos de JustLash. Segmento: {segment}.",
        knowledge=KnowledgeBridge()
    )


def _build_remarketing() -> Agent:
    """Construye el agente Remarketing."""
    return Agent(
        name="Reactivadora Diamante",
        agent_type=AgentType.REMARKETING,
        model=MODELS[AgentType.REMARKETING],
        system_prompt=REMARKETING_SYSTEM_PROMPT,
        temperature=0.8,
        max_tokens=400,
        description="Re-engancha leads fríos con hook emocional y prueba social.",
    )


def get_agent(agent_type: AgentType, segment: str = "1A") -> Agent:
    """Obtiene una instancia del agente solicitado."""
    builders = {
        AgentType.ANFITRION: lambda: _build_anfitrion(),
        AgentType.CONSULTOR: lambda: _build_consultor(),
        AgentType.CLOSER: lambda: _build_closer(segment),
        AgentType.REMARKETING: lambda: _build_remarketing(),
    }

    builder = builders.get(agent_type)
    if builder is None:
        raise ValueError(
            f"Tipo de agente no válido: {agent_type}. "
            f"Opciones: {[t.value for t in AgentType]}"
        )

    return builder()


# Retrocompatibilidad para scripts existentes que usan get_agent(AgentType.QUALIFIER)
def get_agent_for_state(state: LeadState, segment: str = "1A") -> Optional[Agent]:
    """Retrocompatibilidad."""
    if state in TERMINAL_STATES:
        return None
    agent_type = STATE_TO_AGENT.get(state)
    if agent_type is None:
        return None
    return get_agent(agent_type, segment=segment)


# ============================================================================
# ROUTING RULES — Qué agente maneja cada estado
# ============================================================================

STATE_TO_AGENT: dict[LeadState, AgentType] = {
    LeadState.NEW: AgentType.ANFITRION,
    LeadState.QUALIFYING: AgentType.ANFITRION,
    LeadState.QUALIFIED: AgentType.CONSULTOR,
    LeadState.EVALUATING: AgentType.CONSULTOR,
    LeadState.CLOSING: AgentType.CLOSER,
    LeadState.LOST: AgentType.REMARKETING,
    LeadState.REMARKETING: AgentType.REMARKETING,
}

# Estados terminales
TERMINAL_STATES = {LeadState.CONVERTED, LeadState.DEAD}


if __name__ == "__main__":
    print("=" * 60)
    print("💎 Just Lash Academy — Agentes AI Actualizados")
    print("=" * 60)

    for agent_type in AgentType:
        agent = get_agent(agent_type, segment="1A")
        print(f"\n{'─' * 60}")
        print(f"  🤖 {agent.name}")
        print(f"  Tipo: {agent.agent_type.value}")
        print(f"  Modelo: {agent.model}")
        print(f"  Prompt: {len(agent.system_prompt)} caracteres")
