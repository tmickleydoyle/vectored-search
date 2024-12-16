from snippet_printer import get_code_snippet


class QueryEngine:
    """
    Query engine for running questions against the database and printing the results.

    Args:
        db_client (DBClient): The database client.
    """

    def __init__(self, db_client):
        self.db_client = db_client

    def query_and_print(self, question, n_results=1):
        """
        Queries the database with a question and prints the results.

        Args:
            question (str): The question to ask the database.
            n_results (int): The number of results to return.
        """
        results = self.db_client.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        for metadata in results["metadatas"][0]:
            file_path = metadata["file_path"]
            start_row = metadata["start_row"]
            end_row = metadata["end_row"]
            print(f"Question: {question}")
            code_snippet = get_code_snippet(file_path, start_row, end_row)
            print(f"File: {file_path}")
            for line in code_snippet:
                print(line, end="")
            print("---------------")
