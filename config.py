from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Configuration Variables
COLLECTION_NAME = "code_snippets"
DIRECTORY = "/Users/thomasmickley-doyle/repos/language_model"
VECTOR_DB_PATH = "/Users/thomasmickley-doyle/repos/vectordb-search/vector_db"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"  ## "nomic-ai/nomic-embed-text-v1" is pulled from Hugging Face
DEVICE = "cpu"
TRUST_REMOTE_CODE = True
INCLUDED_EXTENSIONS = (".py",)
LANGUAGE_LIB = "build/my-languages.so"

# Correct initialization of the Language object
PYTHON_LANGUAGE = Language(tspython.language())

# Initialize the parser
parser = Parser(PYTHON_LANGUAGE)

LANGUAGE_PARSERS = {
    ".py": parser,
}
