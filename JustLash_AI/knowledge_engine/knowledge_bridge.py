import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_index.faiss"
METADATA_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_metadata.pkl"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

class KnowledgeBridge:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBridge, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        print("Initializing Knowledge Bridge...")
        self.model = SentenceTransformer(MODEL_NAME)
        
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} chunks.")
        else:
            self.index = None
            self.metadata = []
            print("Knowledge index not found.")
        
        self._initialized = True

    def query(self, text, top_k=3):
        if self.index is None:
            return "No hay base de conocimiento disponible."
        
        query_vector = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        if not results:
            return "No se encontró información relevante."
        
        formatted_response = ""
        for r in results:
            formatted_response += f"- {r.get('text', '')} [Fuente: {os.path.basename(r.get('source', ''))}]\n\n"
        
        return formatted_response.strip()

    def get_raw_results(self, text, top_k=3):
        if self.index is None:
            return []
        
        query_vector = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return results

if __name__ == "__main__":
    # Test query
    bridge = KnowledgeBridge()
    print(bridge.query("marketing digital"))
