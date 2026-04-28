"""
agent_router.py — Motor de Routing para Just Lash Academy 💎
=============================================================
Orquesta la conversación con leads: persiste estado, decide qué
agente responde y llama a OpenRouter con el historial correcto.

Componentes:
    - ConversationStore  → Persiste estado por lead en conversations.json
    - OpenRouterClient   → Wrapper de la API de OpenRouter
    - AgentRouter        → Orquestador principal (routing + transiciones)

Uso:
    from agent_router import AgentRouter
    router = AgentRouter()
    response = router.respond(lead_id="maria-01", message="Hola, me interesa el curso")
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from agents import (
    Agent,
    AgentType,
    LeadState,
    Segment,
    get_agent,
    get_agent_for_state,
    STATE_TO_AGENT,
    TERMINAL_STATES,
)

# Cargar variables de entorno desde .env
load_dotenv()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"

DRY_RUN = not bool(OPENROUTER_API_KEY.strip("your-key-here").strip())


# ============================================================================
# DATACLASSES — Modelos de Datos
# ============================================================================

@dataclass
class Lead:
    """Estado completo de un lead en el funnel."""
    lead_id: str
    state: str = LeadState.NEW.value
    segment: str = Segment.UNKNOWN.value
    assigned_agent: str = AgentType.QUALIFIER.value
    history: list = field(default_factory=list)
    remarketing_attempt: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Lead":
        return cls(**data)

    @property
    def lead_state(self) -> LeadState:
        return LeadState(self.state)

    @property
    def lead_segment(self) -> Segment:
        return Segment(self.segment)

    @property
    def is_terminal(self) -> bool:
        return self.lead_state in TERMINAL_STATES


@dataclass
class RouterResponse:
    """Resultado de una interacción del router."""
    lead_id: str
    message: str                    # Respuesta del agente
    agent_name: str                 # Agente que respondió
    agent_type: str                 # Tipo de agente
    state_before: str               # Estado antes de la respuesta
    state_after: str                # Estado después de la respuesta
    segment: str                    # Segmento del lead
    transition_occurred: bool       # ¿Hubo cambio de estado?
    dry_run: bool = False           # ¿Es una respuesta simulada?
    model_used: str = ""            # Modelo de OpenRouter usado
    tokens_used: int = 0            # Tokens consumidos
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# CONVERSATION STORE — Persistencia en JSON
# ============================================================================

class ConversationStore:
    """
    Maneja la persistencia del estado de conversaciones por lead.
    Usa un archivo JSON local como base de datos simple.
    """

    def __init__(self, filepath: Path = CONVERSATIONS_FILE):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Crea el archivo si no existe."""
        if not self.filepath.exists():
            self.filepath.write_text(json.dumps({}, indent=2, ensure_ascii=False))

    def _load(self) -> dict:
        """Carga todos los leads desde el archivo."""
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: dict) -> None:
        """Persiste todos los leads en el archivo."""
        self.filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get(self, lead_id: str) -> Optional[Lead]:
        """Obtiene un lead por ID. Retorna None si no existe."""
        data = self._load()
        lead_data = data.get(lead_id)
        if lead_data is None:
            return None
        return Lead.from_dict(lead_data)

    def get_or_create(self, lead_id: str) -> tuple[Lead, bool]:
        """
        Obtiene un lead existente o crea uno nuevo.

        Returns:
            (lead, created): Lead y bool indicando si fue creado.
        """
        lead = self.get(lead_id)
        if lead is not None:
            return lead, False

        new_lead = Lead(lead_id=lead_id)
        self.save(new_lead)
        return new_lead, True

    def save(self, lead: Lead) -> None:
        """Persiste el estado de un lead."""
        lead.updated_at = datetime.now(timezone.utc).isoformat()
        data = self._load()
        data[lead.lead_id] = lead.to_dict()
        self._save(data)

    def all_leads(self) -> list[Lead]:
        """Retorna todos los leads registrados."""
        data = self._load()
        return [Lead.from_dict(v) for v in data.values()]

    def delete(self, lead_id: str) -> bool:
        """Elimina un lead. Retorna True si existía."""
        data = self._load()
        if lead_id not in data:
            return False
        del data[lead_id]
        self._save(data)
        return True


# ============================================================================
# OPENROUTER CLIENT — Wrapper de la API
# ============================================================================

class OpenRouterClient:
    """
    Wrapper minimalista para la API de OpenRouter.
    Compatible con el estándar OpenAI chat completions.
    """

    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.api_key = api_key
        self.base_url = OPENROUTER_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://justlashacademy.mx",
            "X-OpenRouter-Title": "JustLash Academy Sales Agents",
            "Content-Type": "application/json",
        }

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        retries: int = 2,
    ) -> tuple[str, int]:
        """
        Envía una solicitud de completado a OpenRouter.

        Args:
            messages: Lista de mensajes [{"role": ..., "content": ...}]
            model: Modelo de OpenRouter a usar
            temperature: Creatividad del modelo (0.0-1.0)
            max_tokens: Máximo de tokens en la respuesta
            retries: Intentos en caso de error de red

        Returns:
            (response_text, tokens_used)

        Raises:
            RuntimeError: Si la API devuelve un error después de los reintentos.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return content, tokens

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                last_error = f"HTTP {status}: {e.response.text if e.response else str(e)}"
                if status in (401, 403, 422):
                    break  # No reintentar en errores de auth o payload
                if attempt < retries:
                    time.sleep(1.5 ** attempt)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < retries:
                    time.sleep(1.5 ** attempt)

        raise RuntimeError(f"OpenRouter falló después de {retries + 1} intentos: {last_error}")


# ============================================================================
# TRANSITION DETECTOR — Analiza respuestas para detectar cambios de estado
# ============================================================================

class TransitionDetector:
    """
    Analiza la respuesta del agente para detectar marcadores de estado
    y transiciones en el funnel.
    """

    # Patrones que buscar en la respuesta del agente
    PATTERNS = {
        "classification": r"\[CLASIFICACIÓN:\s*(1A|2A|PENDIENTE)\]",
        "derivar": r"\[DERIVAR:\s*(\w+)\]",
        "estado": r"\[ESTADO:\s*(\w+)\]",
        "intento": r"\[INTENTO:\s*(\d+)/3\]",
    }

    @classmethod
    def detect(cls, response_text: str, current_state: LeadState) -> dict:
        """
        Detecta marcadores en la respuesta y sugiere transiciones.

        Returns:
            dict con keys: new_state, segment, derivar, attempt, markers_found
        """
        result = {
            "new_state": None,
            "segment": None,
            "derivar": None,
            "attempt": None,
            "markers_found": [],
        }

        # Clasificación de segmento (Qualifier)
        m = re.search(cls.PATTERNS["classification"], response_text, re.IGNORECASE)
        if m:
            clasificacion = m.group(1).upper()
            result["markers_found"].append(f"CLASIFICACIÓN:{clasificacion}")
            if clasificacion in ("1A", "2A"):
                result["segment"] = clasificacion
                result["new_state"] = LeadState.QUALIFIED

        # Derivar a otro agente
        m = re.search(cls.PATTERNS["derivar"], response_text, re.IGNORECASE)
        if m:
            result["derivar"] = m.group(1).upper()
            result["markers_found"].append(f"DERIVAR:{result['derivar']}")
            if result["derivar"] == "CLOSER" and result["new_state"] is None:
                result["new_state"] = LeadState.QUALIFIED

        # Estado explícito del Closer o Remarketing
        m = re.search(cls.PATTERNS["estado"], response_text, re.IGNORECASE)
        if m:
            estado_str = m.group(1).upper()
            result["markers_found"].append(f"ESTADO:{estado_str}")
            try:
                result["new_state"] = LeadState(estado_str.lower())
            except ValueError:
                pass  # Estado desconocido, ignorar

        # Intento de remarketing
        m = re.search(cls.PATTERNS["intento"], response_text)
        if m:
            result["attempt"] = int(m.group(1))
            result["markers_found"].append(f"INTENTO:{result['attempt']}/3")

        return result

    @classmethod
    def clean_response(cls, response_text: str) -> str:
        """Elimina los marcadores internos de la respuesta visible al lead."""
        cleaned = response_text
        for pattern in cls.PATTERNS.values():
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()


# ============================================================================
# AGENT ROUTER — Orquestador Principal
# ============================================================================

class AgentRouter:
    """
    Orquestador principal del sistema multi-agente.

    Responsabilidades:
    - Leer el estado actual del lead
    - Seleccionar el agente correcto
    - Construir el contexto para OpenRouter
    - Detectar transiciones de estado en la respuesta
    - Persistir el estado actualizado
    """

    def __init__(self, dry_run: bool = None):
        self.store = ConversationStore()
        self.client = OpenRouterClient()
        self.detector = TransitionDetector()
        # dry_run auto-detectado si no se especifica
        self.dry_run = dry_run if dry_run is not None else DRY_RUN

        if self.dry_run:
            print("⚠️  MODO DRY-RUN: No se llamará a OpenRouter. Respuestas simuladas.")

    def respond(self, lead_id: str, message: str) -> RouterResponse:
        """
        Punto de entrada principal. Recibe un mensaje de un lead y
        retorna la respuesta del agente correcto.

        Args:
            lead_id: Identificador único del lead (ej: número de WhatsApp)
            message: Mensaje enviado por el lead

        Returns:
            RouterResponse con la respuesta y metadata de la interacción
        """
        # 1. Obtener o crear el lead
        lead, is_new = self.store.get_or_create(lead_id)

        if lead.is_terminal:
            return RouterResponse(
                lead_id=lead_id,
                message=f"Este lead está en estado terminal: {lead.state}. No se procesa más.",
                agent_name="Sistema",
                agent_type="system",
                state_before=lead.state,
                state_after=lead.state,
                segment=lead.segment,
                transition_occurred=False,
            )

        state_before = lead.state

        # 2. Seleccionar el agente correcto
        agent = get_agent_for_state(lead.lead_state, segment=lead.segment)
        if agent is None:
            agent = get_agent(AgentType.QUALIFIER)

        # 3. Actualizar historial con el mensaje del lead
        lead.history.append({"role": "user", "content": message})
        lead.state = (
            LeadState.QUALIFYING.value
            if lead.lead_state == LeadState.NEW
            else lead.state
        )
        lead.assigned_agent = agent.agent_type.value

        # 4. Obtener respuesta
        response_text, tokens = self._get_response(agent, lead)

        # 5. Detectar transiciones
        transitions = self.detector.detect(response_text, lead.lead_state)
        transition_occurred = False

        if transitions["segment"] and lead.segment == Segment.UNKNOWN.value:
            lead.segment = transitions["segment"]

        if transitions["new_state"] and transitions["new_state"] != lead.lead_state:
            lead.state = transitions["new_state"].value
            transition_occurred = True

        if transitions["attempt"] is not None:
            lead.remarketing_attempt = transitions["attempt"]

        # 6. Limpiar marcadores internos de la respuesta
        clean_response = self.detector.clean_response(response_text)

        # 7. Agregar respuesta del agente al historial
        lead.history.append({"role": "assistant", "content": clean_response})

        # 8. Persistir estado actualizado
        self.store.save(lead)

        return RouterResponse(
            lead_id=lead_id,
            message=clean_response,
            agent_name=agent.name,
            agent_type=agent.agent_type.value,
            state_before=state_before,
            state_after=lead.state,
            segment=lead.segment,
            transition_occurred=transition_occurred,
            dry_run=self.dry_run,
            model_used=agent.model if not self.dry_run else "dry-run",
            tokens_used=tokens,
        )

    def _get_response(self, agent: Agent, lead: Lead) -> tuple[str, int]:
        """Obtiene la respuesta del agente (real o simulada)."""
        if self.dry_run:
            return self._dry_run_response(agent, lead), 0

        messages = agent.build_messages(lead.history)
        params = agent.to_api_params()

        return self.client.complete(
            messages=messages,
            model=params["model"],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )

    def _dry_run_response(self, agent: Agent, lead: Lead) -> str:
        """Genera una respuesta simulada para dry-run."""
        last_user_msg = next(
            (m["content"] for m in reversed(lead.history) if m["role"] == "user"),
            "",
        )
        templates = {
            AgentType.QUALIFIER: (
                f"¡Hola! 😊 Soy parte del equipo de Just Lash Academy, \n"
                f"a media cuadra del Metro Balbuena. ¿Tenés experiencia \n"
                f"previa con extensiones de pestañas o estás empezando desde cero?\n"
                f"[CLASIFICACIÓN: PENDIENTE]"
            ),
            AgentType.CLOSER: (
                f"¡Perfecto! 💎 Te cuento que tenemos solo 6 lugares por \n"
                f"grupo y quedan 2 disponibles para este mes. Con $1,000 de \n"
                f"apartado asegurás tu lugar. Una vez que terminés el curso, \n"
                f"con 2 clientas ya recuperás la inversión completa. ¿Te agendo?\n"
                f"[ESTADO: CLOSING]"
            ),
            AgentType.REMARKETING: (
                f"¡Hola! 😊 Solo quería ver si te quedó alguna duda. \n"
                f"Esta semana se inscribieron 3 alumnas nuevas y una me dijo \n"
                f"que casi no se animaba, jaja. ¿Qué te frenó?\n"
                f"[INTENTO: 1/3]"
            ),
        }
        return templates.get(agent.agent_type, "[DRY-RUN: Sin respuesta]")

    def reset_lead(self, lead_id: str) -> bool:
        """Reinicia el estado de un lead (útil para testing)."""
        return self.store.delete(lead_id)

    def get_lead_status(self, lead_id: str) -> Optional[Lead]:
        """Retorna el estado actual de un lead."""
        return self.store.get(lead_id)

    def list_leads(self) -> list[Lead]:
        """Lista todos los leads registrados."""
        return self.store.all_leads()
