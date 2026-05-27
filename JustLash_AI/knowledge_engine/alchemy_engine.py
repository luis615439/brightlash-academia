import os
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv("/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SALES_RESOURCES_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/RECURSOS_VENTAS/"

def distill_and_convert(file_path, topic, content_type="guion"):
    print(f"🔮 INICIANDO ALQUIMIA DE ({content_type.upper()}) PARA: {os.path.basename(file_path)}")
    
    # Directorios de salida
    LESSONS_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/MICRO_LECCIONES/"
    RESOURCES_DIR = "/Volumes/IA_LAB_DAT/PORTAL_DE_CRISTAL/RECURSOS_VENTAS/"
    
    # Mockup de lectura de contenido (En producción, usaría PyPDF2 o similar)
    # Por ahora, usamos el nombre del archivo y el tópico para la alquimia creativa.
    filename = os.path.basename(file_path)
    
    if content_type == "leccion":
        if topic == "BELLEZA_Y_LASHES":
            task_desc = f"Crea una MICRO-LECCIÓN de maestría técnica basada en '{filename}'. Enfócate en el arte de las pestañas y la excelencia operativa."
            format_desc = "Título Diamante, El Secreto del Set Perfecto, Aplicación Técnica (paso a paso o consejo Pro) y Cierre de Autoridad."
        else:
            task_desc = f"Crea una MICRO-LECCIÓN diaria de alto impacto basada en '{filename}' sobre el tema '{topic}'."
            format_desc = "Título llamativo, Idea Fuerza, Aplicación Práctica (en el mundo de las pestañas) y un Cierre Inspirador."
    else:
        task_desc = f"Transforma las ideas de '{filename}' sobre '{topic}' en un guion de ventas de alto impacto."
        format_desc = "Hook de Autoridad, Valor Diferencial (Arquitectas de la Belleza), Técnica de Cierre (Apartado $1,000) y Urgencia."

    prompt = f"""
    Eres Antigravity, un Senior Architect y mentor con 15 años de experiencia en el ecosistema JustLash.
    Tu tono es apasionado, directo y técnico, pero desde un lugar de cuidado paternal/mentor (quieres que tus alumnas brillen).
    
    CONTEXTO:
    Estamos en la 'Misión Diamante'. Tu objetivo es destilar sabiduría pura para las 'Arquitectas de la Belleza'.
    
    TAREA:
    1. {task_desc}
    2. Estándar: Diamante / Lujo Boutique.
    
    FORMATO DE SALIDA (Markdown):
    # {content_type.capitalize()}: {filename.split('.')[0]}
    {format_desc}
    
    ---
    *Destilado con precisión por Antigravity para Just Lash Academy.*
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://justlash.ai",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        result = response.json()
        script_content = result['choices'][0]['message']['content']
        
        # Guardar en el directorio correspondiente
        if content_type == "leccion":
            output_filename = f"LECCION_{filename.split('.')[0]}.md"
            output_path = os.path.join(LESSONS_DIR, output_filename)
        else:
            output_filename = f"GUION_{filename.split('.')[0]}.md"
            output_path = os.path.join(RESOURCES_DIR, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        print(f"✅ ALQUIMIA COMPLETADA: {output_filename} guardado.")
        return True
    except Exception as e:
        print(f"❌ Error en Alquimia: {e}")
        return False

if __name__ == "__main__":
    # Prueba manual
    distill_and_convert("Estrategias_de_Persuasion.pdf", "PNL")
