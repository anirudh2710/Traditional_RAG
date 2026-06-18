
from source.data_loader import load_all_documents
from source.vectorstore import FaissVectorStore
from source.search import RAGSearch

if __name__ == "__main__":
    rag = RAGSearch()
    query = "Can I deploy a cloud container using a wildcard asterisk role permission?"
    print(rag.search_and_summarize(query, top_k=3))
     