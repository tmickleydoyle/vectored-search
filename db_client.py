from chromadb import HttpClient
from chromadb.utils import embedding_functions
from config import MODEL_NAME, DEVICE, TRUST_REMOTE_CODE


class DBClient:
    """
    DBClient class for interacting with the VectorDB database.
    """

    def __init__(self, collection_name):
        """
        SentenceTransformerEmbeddingFunction is used to encode the text into embeddings.

        Args:
            collection_name (str): The name of the collection in the database.
        """
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME, device=DEVICE, trust_remote_code=TRUST_REMOTE_CODE
        )
        self.client = HttpClient(
            host="localhost", port=8000
        )  # Docker container running on ChromaDB
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=ef
        )
        # self.collection.delete()

    def upsert_document(self, document, doc_id, metadata):
        """
        Upserts a document into the database.

        Args:
            document (str): The document to upsert.
            doc_id (str): The ID of the document.
            metadata (dict): The metadata associated with the document.
        """
        document = f"{document}"
        self.collection.upsert(documents=[document], ids=[doc_id], metadatas=[metadata])

    def query(self, query_texts, n_results, include):
        """
        Queries the database with a list of query texts and returns the results.

        Args:
            query_texts (list): The list of query texts.
            n_results (int): The number of results to return.
            include (list): The list of items to include in the results.
        """
        return self.collection.query(
            query_texts=query_texts, n_results=n_results, include=include
        )
