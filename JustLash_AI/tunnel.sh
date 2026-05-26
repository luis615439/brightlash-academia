#!/bin/bash
# tunnel.sh — Exponer la API local de JustLash a n8n en la nube
#

PORT=8000

# Colores para la terminal
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${GREEN}💎 Exponiendo Diamond Vault API & AgentRouter local (Puerto $PORT) ${RESET}"

if command -v ngrok &> /dev/null; then
    echo -e "${GREEN}✓ ngrok detectado. Iniciando túnel...${RESET}"
    echo -e "${YELLOW}Endpoint de Chat para n8n será: http://<tu-subdominio>.ngrok-free.app/api/chat${RESET}"
    ngrok http $PORT
else
    echo -e "${RED}✗ ngrok no está instalado en el sistema.${RESET}"
    echo -e "${YELLOW}¿Querés usar el túnel temporal gratuito vía SSH (localhost.run) como alternativa? (s/n)${RESET}"
    read -r respuesta
    if [ "$respuesta" = "s" ] || [ "$respuesta" = "S" ] || [ -z "$respuesta" ]; then
        echo -e "${GREEN}Iniciando túnel seguro vía localhost.run...${RESET}"
        echo -e "${YELLOW}Copia la URL '.lvh.me' o '.localhost.run' provista abajo y usala en n8n Cloud:${RESET}"
        ssh -R 80:localhost:$PORT nokey@localhost.run
    else
        echo -e "${YELLOW}Para instalar ngrok en mac:${RESET}"
        echo -e "  brew install --cask ngrok"
        echo -e "Y luego registrá tu authtoken con:"
        echo -e "  ngrok config add-authtoken <tu-token>"
    fi
fi
