import os
import sqlite3
import pypdf
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
INDEX_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_index.faiss"
METADATA_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_metadata.pkl"

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            # Try pypdf first (robust)
            try:
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e_pdf:
                print(f"pypdf failed for {file_path}, trying pdfplumber: {e_pdf}")
                # Fallback to pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
        elif ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    
    # Limpiar caracteres no imprimibles y bytes nulos
    clean_text = "".join(char for char in text if char.isprintable() or char in "\n\r\t")
    return clean_text.strip()

def chunk_text(text, filename):
    chunks = []
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "source": filename
            })
    return chunks

def index_files():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, current_path FROM files WHERE text_extracted = 0")
    files_to_process = cursor.fetchall()
    
    all_chunks = []
    
    for file_id, filename, file_path in files_to_process:
        print(f"Processing {filename}...")
        text = extract_text(file_path)
        if text:
            chunks = chunk_text(text, filename)
            all_chunks.extend(chunks)
            cursor.execute("UPDATE files SET text_extracted = 1 WHERE id = ?", (file_id,))
    
    if not all_chunks:
        print("No new chunks to index.")
        conn.close()
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts)
    
    # Load or create index
    dimension = embeddings.shape[1]
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []
    
    index.add(np.array(embeddings).astype('float32'))
    metadata.extend(all_chunks)
    
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Mark as indexed in DB
    for file_id, _, _ in files_to_process:
        cursor.execute("UPDATE files SET vector_indexed = 1 WHERE id = ?", (file_id,))
    
    conn.commit()
    conn.close()
    print("Indexing complete.")

if __name__ == "__main__":
    index_files()
