import warnings
import argparse
from db_client import DBClient
from code_loader import load_code_files
from query_engine import QueryEngine
from rag import rag
from config import COLLECTION_NAME, DIRECTORY

warnings.filterwarnings("ignore")


def train():
    """
    Load code snippets and upsert them into the database.
    """
    db_client = DBClient(COLLECTION_NAME)
    python_code = load_code_files(DIRECTORY)

    print("Training...")

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


def generate(query, verbose=False):
    """
    Generate a response for the given query.
    """
    db_client = DBClient(COLLECTION_NAME)
    query_engine = QueryEngine(db_client)
    return query_engine.query(query, n_results=3, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or generate responses for a query."
    )

    parser.add_argument("--train", type=str, help="Whether to train the model")
    parser.add_argument(
        "--query-examples",
        type=str,
        help="Query the vector database with example questions",
    )
    parser.add_argument(
        "--rag-generate",
        type=str,
        help="Query the vector database and pass the question and response to the LLM",
    )

    args = parser.parse_args()

    if args.train == "True":
        train()

    if (
        args.query_examples == "True"
    ):  # Fixed argument name from args.query to args.query_examples
        # Example queries
        questions = [
            "What is the context length for the model?",
            "GPTLanguageModel",
            "How can I load the decoder from the tokenizer?",
            "What is the default device for the model?",
            "Where can I find the byte pair encoder?",
            "block_size",
        ]

        for question in questions:
            generate(question, verbose=True)

    if args.rag_generate:
        vector_response = generate(args.rag_generate, verbose=False)
        response_string = ""
        sources = "Source Files:\n"
        for documents, metadatas in zip(
            vector_response["documents"], vector_response["metadatas"]
        ):
            for document, metadata in zip(documents, metadatas):
                response_string += document
                sources += f"{metadata['file_path']}:{metadata['start_row']}-{metadata['end_row']}\n"

        reg_response = rag(args.rag_generate, response_string)
        print(reg_response["response"])
        print(sources)

    if not any([args.train, args.query_examples, args.rag_generate]):
        print("No arguments provided. Use --help to see available options.")


if __name__ == "__main__":
    main()
