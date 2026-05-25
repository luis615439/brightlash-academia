# Superhuman Core Framework 💎

Este es un framework base limpio y modular para aplicaciones multiproyecto. Proporciona una arquitectura desacoplada con un backend en **FastAPI**, un frontend en **React + TypeScript + Vite + Tailwind CSS** con navegación modular, y soporte para automatización mediante webhooks en **n8n**.

## Estructura del Proyecto

```text
/Superhuman_Core_Framework
  /backend
    ├── main.py                # Servidor FastAPI con endpoints genéricos y autenticación
    ├── requirements.txt       # Dependencias básicas de Python
    └── .env.example           # Variables de entorno de muestra
  /frontend
    ├── package.json           # Dependencias de React y scripts de desarrollo
    ├── vite.config.ts         # Configuración del empaquetador Vite
    ├── tailwind.config.js     # Configuración de estilos Tailwind CSS
    ├── index.html             # Entrada de la App
    └── /src
        ├── main.tsx           # Entrypoint de React
        ├── index.css          # Estilos globales y tokens de diseño
        ├── App.tsx            # Enrutador principal y layout
        └── /components
            ├── SideMenu.tsx   # Menú de navegación dinámico y modular
            ├── ThemeToggle.tsx# Cambiador de temas (Claro/Oscuro)
            ├── GenericForm.tsx# Formulario genérico con envío a webhook de n8n
            └── GenericMetricsAuditor.tsx # Auditor financiero y de métricas visual
  /workflows
    └── n8n_template.json      # Plantilla de webhook, conmutación y alertas para n8n
  └── README.md                # Esta guía
```

## Requisitos Previos

- **Node.js** (v18 o superior) y **npm**
- **Python** (v3.10 o superior) y **pip**

---

## Guía Rápida de Inicio

### 1. Iniciar el Backend (FastAPI)

1. Ingresa a la carpeta `backend`:
   ```bash
   cd backend
   ```
2. Crea e inicia un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configura el archivo `.env`:
   ```bash
   cp .env.example .env
   ```
5. Inicia el servidor de desarrollo:
   ```bash
   python main.py
   ```
   *El servidor correrá en `http://localhost:8000`.*

### 2. Iniciar el Frontend (React + Vite)

1. Ingresa a la carpeta `frontend`:
   ```bash
   cd ../frontend
   ```
2. Instala las dependencias:
   ```bash
   npm install
   ```
3. Inicia el servidor de desarrollo:
   ```bash
   npm run dev
   ```
   *El portal se abrirá en `http://localhost:5173`.*

---

## Checklist de Clonación para Nuevos Proyectos

- [ ] Modificar `SUPERHUMAN_API_KEY` en `backend/.env` para producción.
- [ ] Registrar tus nuevos componentes en `frontend/src/App.tsx` agregando los casos al `switch (activeApp)`.
- [ ] Añadir los accesos de menú correspondientes en el array de `apps` dentro de `frontend/src/components/SideMenu.tsx`.
- [ ] Importar el JSON de `workflows/n8n_template.json` en n8n para configurar las integraciones.
