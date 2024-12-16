import argparse
from db_client import DBClient
from code_loader import load_code_files
from query_engine import QueryEngine
from config import COLLECTION_NAME, DIRECTORY


def train():
    """
    Load code snippets and upsert them into the database.
    """
    db_client = DBClient(COLLECTION_NAME)
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
    print("Training completed.")


def generate(query):
    """
    Generate a response for the given query.
    """
    db_client = DBClient(COLLECTION_NAME)
    query_engine = QueryEngine(db_client)
    query_engine.query_and_print(query)


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or generate responses for a query."
    )

    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Whether to generate a response"
    )
    parser.add_argument(
        "--query", type=str, help="The query string for generating a response"
    )

    args = parser.parse_args()

    if args.train:
        train()

    if args.generate:
        # Example queries
        questions = [
            "What is the context length for the model?",
            "GPTLanguageModel",
            "How can I load the decoder from the tokenizer?",
            "What is the default device for the model?",
            "Where can I find the byte pair encoder?",
        ]

        for question in questions:
            generate(question)


if __name__ == "__main__":
    main()
