import sys
import os
import json

# Asegurar compatibilidad de colores en terminal
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GOLD    = "\033[38;5;178m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    DIM     = "\033[2m"

def print_banner():
    print(f"{C.GOLD}{C.BOLD}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.GOLD}{C.BOLD}║   💎  SIMULACIÓN: FLUJO DE CALIFICACIÓN Y DOSIFICACIÓN       ║{C.RESET}")
    print(f"{C.GOLD}{C.BOLD}║   Arnés de Verificación JustLash (Hermes Engine v2.0)         ║{C.RESET}")
    print(f"{C.GOLD}{C.BOLD}╚══════════════════════════════════════════════════════════════╝{C.RESET}\n")

def print_turn(sender, role, message, principle=None):
    if sender == "Prospecto (Camila)":
        print(f"📱 {C.BOLD}{C.CYAN}Camila (Lead):{C.RESET} {message}")
    else:
        print(f"\n{C.GOLD}🤖 {C.BOLD}{sender} [{role}]{C.RESET}")
        if principle:
            print(f"   {C.DIM}↳ Principio Cialdini: {principle}{C.RESET}")
        print(f"   {C.GREEN}\" {message} \"{C.RESET}")
    print("─" * 70)

def main():
    print_banner()

    # Turno 1: El Anfitrión
    print_turn(
        "Prospecto (Camila)", 
        "User", 
        "Hola, me interesa información sobre el curso básico de pestañas de este mes."
    )
    
    print_turn(
        "El Anfitrión",
        "Primer Contacto - Simpatía y Reciprocidad",
        "¡Hola, Camila! ✨ Qué gusto que nos escribas a JUST LASH. Soy Sofía y estoy aquí para guiarte. \n   Antes de pasarte todos los detalles, te comparto de regalo nuestra Guía Rápida de Visajismo para que vayas conociendo el arte de la mirada. 💖\n   Cuéntame un poco para conocerte: ¿Este sería tu primer paso en el mundo de la belleza o ya tienes algo de experiencia en el área?",
        "Reciprocidad (regala guía) + Simpatía (personalización por nombre)"
    )

    # Turno 2: El Consultor
    print_turn(
        "Prospecto (Camila)",
        "User",
        "Es mi primer paso, empiezo totalmente desde cero."
    )

    print_turn(
        "El Consultor",
        "Calificación de Perfil - Autoridad y Prueba Social",
        "¡Qué gran paso estás por dar, Camila! Como bien dicen nuestras más de 5,000 graduadas, el mejor momento para empezar fue ayer, el segundo mejor es hoy. 🏆\n   Te cuento: Nuestro Curso Inicial está diseñado para principiantes absolutas desde cero. PERO ojo, no es un curso 'hobby' de fin de semana para pasar el rato; es el inicio formal de una carrera profesional de alto rendimiento en el mundo lashista.\n   ¿Estás buscando aprender esto para iniciar tu propio negocio de belleza o para aplicarlo de forma casual?",
        "Autoridad (trayectoria) + Prueba Social (5,000 graduadas) + Filtro de Compromiso (No es un Hobby)"
    )

    # Turno 3: El Closer
    print_turn(
        "Prospecto (Camila)",
        "User",
        "Quiero poner mi propio estudio de pestañas en casa, tomármelo en serio y vivir de esto."
    )

    print_turn(
        "El Closer",
        "Cierre y Filtros del Arnés de Verificación - Escasez y Compromiso",
        "Me encanta esa determinación, Camila. Ese es exactamente el perfil de alumna que buscamos en la academia.\n   La inversión para el Curso Inicial es de $5,500 MXN en Metro Balbuena. Ahora, como buscamos formar profesionales de élite, debes conocer nuestras dos reglas de oro:\n\n"
        "   1. 🎓 PROGRESIÓN OBLIGATORIA: Te enseñaremos la técnica clásica perfecta (aislamiento, peso y salud natural) como base obligatoria. Solo dominando esto al 100% tendrás derecho a cursar los siguientes niveles avanzados de la academia (Cat Eye, Doll Eye, Volúmenes Híbridos).\n"
        "   2. 🛠️ CLÁUSULA DE PRÁCTICA: El curso no termina en el aula. Para recibir tu diploma oficial con aval, te comprometes a entregar evidencia fotográfica de prácticas en modelos reales las semanas posteriores. Si no practicas, no hay diploma.\n"
        "   3. ⏱️ POLÍTICA MILITAR: Cero tolerancia a retardos para no interrumpir el aprendizaje del grupo.\n\n"
        "   Sabiendo esto, ¿estás lista para comprometerte a practicar con modelos post-curso y respetar la puntualidad para ganarte tu certificación?",
        "Coherencia y Compromiso (el prospecto se compromete públicamente a cumplir las reglas de excelencia)"
    )

    # Turno 4: Aprobación del Arnés de Verificación
    print_turn(
        "Prospecto (Camila)",
        "User",
        "Sí, me comprometo totalmente con las prácticas y la puntualidad. Quiero capacitarme con el Estándar Diamante."
    )

    # Verify Harness JSON Output
    print(f"\n{C.GOLD}{C.BOLD}⚡ [VERIFY HARNESS: EVALUACIÓN DE GATE RULES PARA SUPABASE]{C.RESET}")
    harness_output = {
        "lead_status": "PENDIENTE_APROBACION",
        "perfil_aspirante": "Principiante absoluto",
        "compromiso_practica": True,
        "motivo_excelencia": "Quiero poner mi propio estudio de pestañas en casa y tomármelo en serio.",
        "disponibilidad_estricta": True,
        "resumen_verificacion": "Aspirante comprometida, busca autoempleo formal, acepta la cláusula de práctica post-curso y el prerrequisito de clásica."
    }
    print(C.YELLOW + json.dumps(harness_output, indent=2, ensure_ascii=False) + C.RESET)
    print(f"{C.GREEN}{C.BOLD}✅ LEAD APROBADO: Perfil apto para el Curso Básico. Guardando en CRM.{C.RESET}\n")

    # Closer: Cierre final de apartado
    print_turn(
        "El Closer",
        "Cierre de Apartado",
        "¡Excelente, Camila! Perfil aprobado para la próxima generación. Nos quedan solo 2 lugares disponibles para la fecha de este mes con el Kit Premium y los bonus incluidos.\n   Puedes asegurar tu lugar hoy mismo con un apartado de $1,000 MXN. ¿Prefieres hacer el apartado por transferencia bancaria o te genero un link de pago?",
        "Escasez (2 lugares) + Acción Directa"
    )

if __name__ == "__main__":
    main()
