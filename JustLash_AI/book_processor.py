import os
import subprocess
import json
from pathlib import Path
from agent_router import OpenRouterClient

def get_text_from_docx(file_path):
    """Convierte un docx a texto usando textutil (solo Mac)."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', str(file_path), '-stdout'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error al leer {file_path}: {e}")
        return None

def process_books():
    client = OpenRouterClient()
    # Cargamos la config para el prompt prefix del creador de contenido
    config_path = Path(__file__).parent / "agentes_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    creator_config = config['agents']['content_creator']
    
    books_dir = Path("/Users/joseluis/Downloads/ideas contenidolibros")
    output_dir = Path(__file__).parent / "instagram_content"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Iniciando procesamiento de libros con {creator_config['model']}...")
    
    for docx_file in books_dir.glob("*.docx"):
        print(f"📖 Procesando: {docx_file.name}")
        content = get_text_from_docx(docx_file)
        
        if not content:
            continue
            
        prompt = f"{creator_config['prompt_prefix']}\n\nCONTENIDO DEL LIBRO:\n{content}\n\nTAREA: Creá 3 variantes de copys para Instagram Stories (Hook + Valor + CTA) basados en esta info."
        
        try:
            response, tokens = client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=creator_config['model'],
                temperature=creator_config['temperature'],
                max_tokens=creator_config['max_tokens']
            )
            
            output_file = output_dir / f"{docx_file.stem}_content.txt"
            output_file.write_text(response, encoding='utf-8')
            print(f"✅ Contenido generado en: {output_file.name} ({tokens} tokens)")
            
        except Exception as e:
            print(f"❌ Error al procesar {docx_file.name} con Gemma 4: {e}")

if __name__ == "__main__":
    process_books()
