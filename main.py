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
    for filename, snippet, row_number, file_path in python_code:
        db_client.upsert_document(
            document=snippet,
            doc_id=f"{filename}-{row_number}",
            metadata={"file_path": file_path, "row_number": row_number},
        )

    # Example queries
    questions = [
        "What is the context length for the model?",
        "What is the learning rate used in the model?",
        "What is the dropout rate used in the model?",
        "How is the training data loaded into the model?",
        "What is the Byte Pair Encoding tokenizer?",
        "GPTLanguageModel",
        "How can I load the decoder from the tokenizer?",
    ]

    for question in questions:
        query_engine.query_and_print(question)

if __name__ == "__main__":
    main()
