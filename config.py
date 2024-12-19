from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Configuration Variables
COLLECTION_NAME = "code_snippets"  # Name of the collection in ChromaDB
DIRECTORY = "/Users/thomasmickley-doyle/repos/language-model"  # Directory to search for code snippets
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"  # Pulled from Hugging Face
DEVICE = "cpu"  # Device to run the model on
TRUST_REMOTE_CODE = True  # Model setting to pull from Hugging Face
INCLUDED_EXTENSIONS = (".py",)  # File extensions to include in the search

# Correct initialization of the Language object
PYTHON_LANGUAGE = Language(tspython.language())

# Initialize the parser
parser = Parser(PYTHON_LANGUAGE)

LANGUAGE_PARSERS = {
    ".py": parser,
}
