"""
simulator.py — Simulador de WhatsApp para Just Lash Academy 💎
==============================================================
CLI interactivo que simula una conversación de WhatsApp con el
sistema de agentes. Ideal para probar el flujo completo antes
de conectar a WhatsApp real.

Uso:
    python simulator.py                    # Nuevo lead aleatorio
    python simulator.py --lead maria-01    # Lead específico
    python simulator.py --reset maria-01   # Reiniciar lead
    python simulator.py --status           # Ver todos los leads
    python simulator.py --dry-run          # Forzar modo sin API
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ─── Compatibilidad de colores en terminal ────────────────────────────────
try:
    import shutil
    _cols = shutil.get_terminal_size().columns
except Exception:
    _cols = 80

class C:
    """Códigos ANSI para colores en terminal."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GOLD    = "\033[38;5;178m"      # #C9A96E aprox
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BGDARK  = "\033[48;5;232m"      # Fondo oscuro Estándar Diamante

def gold(text: str) -> str:
    return f"{C.GOLD}{C.BOLD}{text}{C.RESET}"

def dim(text: str) -> str:
    return f"{C.DIM}{C.GRAY}{text}{C.RESET}"

def bold(text: str) -> str:
    return f"{C.BOLD}{text}{C.RESET}"

def green(text: str) -> str:
    return f"{C.GREEN}{text}{C.RESET}"

def red(text: str) -> str:
    return f"{C.RED}{text}{C.RESET}"

def yellow(text: str) -> str:
    return f"{C.YELLOW}{text}{C.RESET}"

def cyan(text: str) -> str:
    return f"{C.CYAN}{text}{C.RESET}"

# ─────────────────────────────────────────────────────────────────────────────

from agent_router import AgentRouter, RouterResponse
from agents import AgentType, LeadState


# ============================================================================
# UI HELPERS
# ============================================================================

SEPARATOR = gold("─" * min(_cols, 65))

BANNER = f"""
{gold("╔══════════════════════════════════════════════════════════════╗")}
{gold("║")}  {bold("💎  Just Lash Academy — Simulador de Agentes AI")}            {gold("║")}
{gold("║")}  {dim("Sistema de ventas inteligente powered by OpenRouter")}         {gold("║")}
{gold("╚══════════════════════════════════════════════════════════════╝")}
"""

AGENT_ICONS = {
    "qualifier":   "🟢",
    "closer":      "🔴",
    "remarketing": "🟡",
    "system":      "⚙️ ",
}

STATE_COLORS = {
    "new":          lambda s: dim(s),
    "qualifying":   lambda s: cyan(s),
    "qualified":    lambda s: yellow(s),
    "closing":      lambda s: yellow(s),
    "converted":    lambda s: green(s),
    "lost":         lambda s: red(s),
    "remarketing":  lambda s: yellow(s),
    "dead":         lambda s: red(s),
}


def colorize_state(state: str) -> str:
    fn = STATE_COLORS.get(state, lambda s: s)
    return fn(state.upper())


def print_banner() -> None:
    print(BANNER)


def print_separator() -> None:
    print(f"\n{SEPARATOR}\n")


def print_agent_response(response: RouterResponse) -> None:
    """Imprime la respuesta del agente con formato visual."""
    icon = AGENT_ICONS.get(response.agent_type, "🤖")
    state_str = colorize_state(response.state_after)

    print(f"\n{icon}  {gold(response.agent_name)}")

    # Metadata de la interacción
    meta_parts = [f"Estado: {state_str}"]
    if response.segment != "unknown":
        meta_parts.append(f"Segmento: {bold(response.segment)}")
    if response.transition_occurred:
        meta_parts.append(green("⚡ Transición detectada"))
    if response.dry_run:
        meta_parts.append(yellow("[DRY-RUN]"))
    elif response.model_used:
        meta_parts.append(dim(f"↳ {response.model_used}"))
    if response.tokens_used > 0:
        meta_parts.append(dim(f"~{response.tokens_used} tokens"))

    print(dim("  " + " · ".join(meta_parts)))
    print()

    # Respuesta del agente — con indentación tipo burbuja de chat
    lines = response.message.split("\n")
    for line in lines:
        if line.strip():
            print(f"  {C.WHITE}{line}{C.RESET}")
        else:
            print()

    print()


def print_lead_status(lead) -> None:
    """Imprime el estado actual de un lead."""
    print(f"\n{gold('📋 Estado del Lead')}")
    print(f"  ID:        {bold(lead.lead_id)}")
    print(f"  Estado:    {colorize_state(lead.state)}")
    print(f"  Segmento:  {bold(lead.segment)}")
    print(f"  Agente:    {lead.assigned_agent}")
    print(f"  Mensajes:  {len(lead.history)}")
    print(f"  Creado:    {dim(lead.created_at[:19].replace('T', ' '))}")
    print()


def print_all_leads(router: AgentRouter) -> None:
    """Lista todos los leads registrados."""
    leads = router.list_leads()
    if not leads:
        print(yellow("\nNo hay leads registrados aún.\n"))
        return

    print(f"\n{gold(f'📊 Leads Registrados ({len(leads)})')}\n")
    print(f"  {'ID':<20} {'Estado':<15} {'Segmento':<10} {'Mensajes':<10}")
    print(f"  {'─'*20} {'─'*15} {'─'*10} {'─'*8}")
    for lead in sorted(leads, key=lambda l: l.updated_at, reverse=True):
        state_col = colorize_state(lead.state)
        print(f"  {lead.lead_id:<20} {state_col:<24} {lead.segment:<10} {len(lead.history):<10}")
    print()


def print_commands() -> None:
    """Muestra los comandos disponibles en el simulador."""
    print(f"\n{gold('Comandos disponibles:')}")
    print(f"  {cyan('/status')}    — Ver estado del lead actual")
    print(f"  {cyan('/reset')}     — Reiniciar este lead desde cero")
    print(f"  {cyan('/leads')}     — Ver todos los leads registrados")
    print(f"  {cyan('/history')}   — Ver historial de conversación")
    print(f"  {cyan('/exit')}      — Salir del simulador")
    print()


# ============================================================================
# SIMULATOR PRINCIPAL
# ============================================================================

def run_simulator(lead_id: str, router: AgentRouter) -> None:
    """
    Loop principal del simulador interactivo.

    Args:
        lead_id: Identificador del lead con quien conversar.
        router: Instancia del AgentRouter configurado.
    """
    print_banner()

    # Info inicial del lead
    lead, is_new = router.store.get_or_create(lead_id)
    if is_new:
        print(green(f"✨ Nuevo lead creado: {bold(lead_id)}"))
    else:
        print(f"🔄 Continuando conversación con: {bold(lead_id)}")
        print_lead_status(lead)

    print(f"\n{dim('Simulá una conversación de WhatsApp con JustLash Academy.')}")
    print_commands()
    print(SEPARATOR)

    # Loop de conversación
    while True:
        try:
            user_input = input(f"\n{bold('Tú')} 📱  ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{dim('Simulación finalizada.')}")
            break

        if not user_input:
            continue

        # ─── Comandos internos ────────────────────────────────────────────
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"\n{dim('Hasta luego 👋')}\n")
            break

        if user_input.lower() == "/status":
            lead = router.get_lead_status(lead_id)
            if lead:
                print_lead_status(lead)
            continue

        if user_input.lower() == "/leads":
            print_all_leads(router)
            continue

        if user_input.lower() == "/reset":
            confirm = input(
                f"{yellow('¿Seguro que querés reiniciar este lead? (s/N): ')}"
            ).strip().lower()
            if confirm == "s":
                router.reset_lead(lead_id)
                lead, _ = router.store.get_or_create(lead_id)
                print(green(f"✅ Lead {lead_id} reiniciado."))
            else:
                print(dim("Cancelado."))
            continue

        if user_input.lower() == "/history":
            lead = router.get_lead_status(lead_id)
            if not lead or not lead.history:
                print(yellow("Sin historial aún."))
            else:
                print(f"\n{gold('📜 Historial de conversación:')}\n")
                for i, msg in enumerate(lead.history, 1):
                    role = bold("Tú") if msg["role"] == "user" else gold("Agente")
                    print(f"  [{i}] {role}: {msg['content'][:120]}...")
            print()
            continue

        # ─── Verificar estado terminal ────────────────────────────────────
        lead = router.get_lead_status(lead_id)
        if lead and lead.is_terminal:
            state_str = colorize_state(lead.state)
            print(f"\n{yellow('⚠️  Este lead está en estado')} {state_str}.")
            print(dim("  Usá /reset para reiniciar o /leads para ver otros leads."))
            continue

        # ─── Llamada al router ────────────────────────────────────────────
        print(dim("\n  Procesando..."))
        try:
            response = router.respond(lead_id=lead_id, message=user_input)
            print_agent_response(response)

            # Alertas de transición
            if response.state_after == LeadState.CONVERTED.value:
                print(green("🎉 ¡CONVERSIÓN EXITOSA! Alumna inscrita."))
                print(dim("  Usá /reset para simular otro lead o /exit para salir."))
            elif response.state_after == LeadState.DEAD.value:
                print(red("💀 Lead descartado después de 3 intentos de remarketing."))
                print(dim("  Usá /reset para reiniciar o /exit para salir."))

        except RuntimeError as e:
            print(red(f"\n❌ Error al conectar con OpenRouter:\n  {e}"))
            print(dim("  Verificá tu API key en el archivo .env"))
        except Exception as e:
            print(red(f"\n❌ Error inesperado: {e}"))
            import traceback
            print(dim(traceback.format_exc()))


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="💎 JustLash Academy — Simulador de Agentes AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python simulator.py                    # Nuevo lead (ID auto-generado)
  python simulator.py --lead 5215500001  # Lead con ID específico
  python simulator.py --reset aria-02   # Reiniciar lead
  python simulator.py --status           # Ver todos los leads
  python simulator.py --dry-run          # Sin llamada real a OpenRouter
        """,
    )
    parser.add_argument(
        "--lead", "-l",
        type=str,
        help="ID del lead (ej: número de WhatsApp sin '+': 5215500001234)",
        default=None,
    )
    parser.add_argument(
        "--reset", "-r",
        type=str,
        metavar="LEAD_ID",
        help="Reinicia el estado de un lead específico",
        default=None,
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Muestra todos los leads registrados y sale",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Modo sin API: respuestas simuladas, no llama a OpenRouter",
        dest="dry_run",
    )

    args = parser.parse_args()

    # Inicializar router
    router = AgentRouter(dry_run=args.dry_run if args.dry_run else None)

    # ─── Modo status ─────────────────────────────────────────────────────
    if args.status:
        print_banner()
        print_all_leads(router)
        return

    # ─── Modo reset ──────────────────────────────────────────────────────
    if args.reset:
        deleted = router.reset_lead(args.reset)
        if deleted:
            print(green(f"✅ Lead '{args.reset}' reiniciado correctamente."))
        else:
            print(yellow(f"⚠️  Lead '{args.reset}' no encontrado."))
        return

    # ─── Modo simulación ─────────────────────────────────────────────────
    lead_id = args.lead
    if not lead_id:
        # Generar ID único basado en timestamp
        lead_id = f"lead-{datetime.now().strftime('%H%M%S')}"

    run_simulator(lead_id=lead_id, router=router)


if __name__ == "__main__":
    main()
