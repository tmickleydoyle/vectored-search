from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from config import VECTOR_DB_PATH, MODEL_NAME, DEVICE

class DBClient:
    def __init__(self, collection_name):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
            device=DEVICE,
        )
        self.client = PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef
        )

    def upsert_document(self, document, doc_id, metadata):
        self.collection.upsert(documents=[document], ids=[doc_id], metadatas=[metadata])

    def query(self, query_texts, n_results, include):
        return self.collection.query(query_texts=query_texts, n_results=n_results, include=include)
