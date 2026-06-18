import os
import json
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Any
from sentence_transformers import SentenceTransformer
from source.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any], data_dir: str = "data"):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        # Reset the index and metadata so we rebuild from scratch cleanly
        self.index = None
        self.metadata = []
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata) if hasattr(chunk, "metadata") and chunk.metadata else {}
            meta["text"] = chunk.page_content
            if "source" in meta:
                meta["source_name"] = os.path.basename(str(meta["source"]))
            metadatas.append(meta)
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        self.save_manifest(data_dir)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def get_data_manifest(self, data_dir: str = "data") -> dict:
        """
        Scan data_dir for all supported files and retrieve their paths and last modified times.
        """
        data_path = Path(data_dir).resolve()
        if not data_path.exists():
            return {}
        
        # Supported extensions matching data_loader.py
        extensions = ['*.pdf', '*.txt', '*.csv', '*.xlsx', '*.docx', '*.json']
        manifest = {}
        for ext in extensions:
            for file in data_path.glob(f'**/{ext}'):
                if file.is_file():
                    try:
                        rel_path = str(file.relative_to(data_path))
                    except ValueError:
                        rel_path = str(file)
                    manifest[rel_path] = file.stat().st_mtime
        return manifest

    def save_manifest(self, data_dir: str = "data"):
        manifest = self.get_data_manifest(data_dir)
        manifest_path = os.path.join(self.persist_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[INFO] Saved index manifest to {manifest_path}")

    def is_out_of_sync(self, data_dir: str = "data") -> bool:
        """
        Check if the cached FAISS index is out of sync with the files in data_dir.
        """
        manifest_path = os.path.join(self.persist_dir, "manifest.json")
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        # If any required file is missing, we are out of sync
        if not (os.path.exists(manifest_path) and os.path.exists(faiss_path) and os.path.exists(meta_path)):
            return True
            
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                saved_manifest = json.load(f)
        except Exception:
            return True
            
        current_manifest = self.get_data_manifest(data_dir)
        return current_manifest != saved_manifest

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from source.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))