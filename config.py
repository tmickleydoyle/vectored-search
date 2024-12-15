import os

# Configuration Variables
COLLECTION_NAME = "mpnet_model_code"
DIRECTORY = "/Users/thomasmickley-doyle/repos/language_model"
VECTOR_DB_PATH = "/Users/thomasmickley-doyle/repos/vectordb-search/vector_db"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1" ## "nomic-ai/nomic-embed-text-v1" is pulled from Hugging Face
DEVICE = "cpu"
TRUST_REMOTE_CODE=True
INCLUDED_EXTENSIONS = (".py",)
