from snippet_printer import pprint_code_snippet

class QueryEngine:
    def __init__(self, db_client):
        self.db_client = db_client

    def query_and_print(self, question, n_results=1):
        results = self.db_client.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        for metadata in results["metadatas"][0]:
            file_path = metadata["file_path"]
            row_number = metadata["row_number"]
            print(f"Question: {question}")
            pprint_code_snippet(file_path, row_number)
            print("---------------")
