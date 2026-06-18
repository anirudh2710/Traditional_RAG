import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from source.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        if self.vectorstore.is_out_of_sync("data"):
            print("[INFO] Vector store is out of sync or missing. Rebuilding index...")
            from source.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs, data_dir="data")
        else:
            self.vectorstore.load()
            
        # Initialize BM25 dynamically from the loaded metadata
        self.corpus = [meta.get("text", "") for meta in self.vectorstore.metadata]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[INFO] Initialized BM25 index with {len(self.corpus)} chunks.")
        
        # Load lightweight Cross-Encoder for re-ranking
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[INFO] Loaded Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[WARNING] Failed to load Cross-Encoder ({e}). Falling back to pure RRF hybrid ranking.")
            self.reranker = None
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def hybrid_search(self, query: str, top_k: int = 15) -> list:
        # 1. Dense retrieval (FAISS)
        dense_results = self.vectorstore.query(query, top_k=top_k)
        
        # 2. Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Dense rank scoring
        for rank, res in enumerate(dense_results):
            idx = int(res["index"])
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (60.0 + rank + 1)
            
        # Sparse rank scoring
        for rank, idx in enumerate(top_bm25_indices):
            score = bm25_scores[idx]
            if score <= 0:  # Skip Chunks with zero term match
                continue
            idx_int = int(idx)
            if idx_int not in rrf_scores:
                rrf_scores[idx_int] = 0.0
            rrf_scores[idx_int] += 1.0 / (60.0 + rank + 1)
            
        # Sort indices by score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_results = []
        for idx in sorted_indices:
            meta = self.vectorstore.metadata[idx]
            final_results.append({
                "index": idx,
                "metadata": meta,
                "rrf_score": rrf_scores[idx]
            })
        return final_results

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Retrieve candidates via Hybrid Search
        candidates = self.hybrid_search(query, top_k=15)
        if not candidates:
            return "No relevant documents found."
            
        # Re-rank candidates using the Cross-Encoder if available, otherwise fall back to RRF rank
        if self.reranker is not None:
            try:
                pairs = [(query, c["metadata"].get("text", "")) for c in candidates]
                rerank_scores = self.reranker.predict(pairs)
                for idx, score in enumerate(rerank_scores):
                    candidates[idx]["score"] = float(score)
                sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
            except Exception as e:
                print(f"[WARNING] Reranking failed ({e}). Falling back to RRF score.")
                sorted_candidates = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)[:top_k]
        else:
            sorted_candidates = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)[:top_k]
        
        # Format grounded context for the LLM
        context_parts = []
        for rank, c in enumerate(sorted_candidates):
            meta = c["metadata"]
            src = meta.get("source_name", "Unknown Document")
            pg = meta.get("page", None)
            page_str = f", Page {pg+1}" if pg is not None else ""
            text = meta.get("text", "")
            context_parts.append(f"--- Document: {src}{page_str} (Rank: {rank+1}) ---\n{text}")
            
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful, professional AI assistant answering questions based strictly on the provided context.

Context:
{context}

User Query: {query}

Instructions:
1. Answer the user query using only the provided context.
2. Provide precise, direct answers.
3. Cite the document names and page numbers in your answer when referencing facts (e.g., "[document_name.pdf, Page X]").
4. If the provided context does not contain the answer, state: "I cannot find the answer in the provided documents." and do not speculate or hallucinate.

Answer:"""
        
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is Satwave?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:\n", summary)