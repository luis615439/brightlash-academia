#!/bin/bash
# lash.sh — Atajo para correr el simulador de JustLash desde cualquier lugar
#
# Uso:
#   ./lash.sh                    # Nuevo lead
#   ./lash.sh --lead maria       # Lead específico
#   ./lash.sh --status           # Ver todos los leads
#   ./lash.sh --reset maria      # Reiniciar lead
#   ./lash.sh --dry-run          # Sin API

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

python3 simulator.py "$@"
