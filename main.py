from db_client import DBClient
from code_loader import load_code_files
from query_engine import QueryEngine
from config import COLLECTION_NAME, DIRECTORY


def main():
    """
    Main function to load code snippets, upsert them into the database, and run example queries.
    """
    db_client = DBClient(COLLECTION_NAME)
    query_engine = QueryEngine(db_client)

    # Load and upsert code snippets
    python_code = load_code_files(DIRECTORY)

    for code in python_code:
        db_client.upsert_document(
            document=code["content"],
            doc_id=f"{code['type']}-{code['start_row']}",
            metadata={
                "type": code["type"],
                "start_row": code["start_row"],
                "end_row": code["end_row"],
                "filename": code["filename"],
                "file_path": code["file_path"],
            },
        )

    # Example queries
    questions = [
        "What is the context length for the model?",
        "GPTLanguageModel",
        "How can I load the decoder from the tokenizer?",
    ]

    for question in questions:
        query_engine.query_and_print(question)


if __name__ == "__main__":
    main()
