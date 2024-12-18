
![6051164](https://github.com/user-attachments/assets/58acd1c9-8aa0-4899-a309-d5a950963104)

## Vector Search Over Code

### About

Uses ChromaDB to index code snippets and perform similarity searches. The code snippets are tokenized and embedded using a transformer model trained on a large codebase, `nomic-embed-text-v1`. The code is parsed with the `tree-sitter` library and the embeddings are generated using the `transformers` library. Natural langauge searches are tokenized and embedded in the same way, and the cosine similarity between the query and the code snippets is used to rank the results. The top `k` results are then returned pretty-printed and displayed in the terminal. To convert this into a RAG model, the code snippets could be used as the `context` and the query could be used as the `question`.

### Project Configs

The majority of the setting for the code are set in the `config.py` file. The `config.py` file contains the following settings:

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Configuration Variables
COLLECTION_NAME = "code_snippets" # Name of the collection in ChromaDB
DIRECTORY = "/Users/thomasmickley-doyle/repos/language_model" # Directory to search for code snippets
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"  # Pulled from Hugging Face
DEVICE = "cpu" # Device to run the model on
TRUST_REMOTE_CODE = True # Model setting to pull from Hugging Face
INCLUDED_EXTENSIONS = (".py",) # File extensions to include in the search

# Correct initialization of the Language object
PYTHON_LANGUAGE = Language(tspython.language())

# Initialize the parser
parser = Parser(PYTHON_LANGUAGE)

LANGUAGE_PARSERS = {
    ".py": parser,
}
```

### Getting Started

1. Clone the repository
2. Run `docker run -p 8000:8000 chromadb/chroma` to start the ChromaDB server
3. Install dependencies: `pip install -r requirements.txt`
4. Run the main script: `python main.py --query-examples False --train True --rag-generate "What does load_data do?"`
5. :pizza:

### Files Structure

```text
project/
│
├── main.py               # Entry point for executing the queries
├── config.py             # Configuration variables
├── db_client.py          # Handles ChromaDB client setup and interactions
├── code_loader.py        # Handles file loading and processing
├── snippet_printer.py    # Pretty-printing snippets
├── query_engine.py       # Encapsulates query logic
└── requirements.txt      # Dependency list
```

### CLI Commands

_Embeddings trained on my [language_model](https://github.com/tmickleydoyle/language_model) repo._

Training on the data
```text
$ python main.py --train True

Training...
Training completed.
```

Generate embeddings for the code snippets. Example questions are in the main.py file.
```text
$ python main.py --query-examples True

Question: What is the context length for the model?
File: /Users/thomasmickley-doyle/repos/language_model/config.py
class Config:
    """
    Configuration class to store hyperparameters and other settings.

    Attributes:
        batch_size: Number of independent sequences to process in parallel.
        block_size: Maximum context length for predictions.
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
        vocab_size: Size of the vocabulary.
        max_iters: Maximum number of training iterations.
        eval_interval: Number of iterations between evaluation runs.
        learning_rate: Learning rate for the optimizer.
        eval_iters: Number of iterations to run evaluation for.
        device: Device to run the model on.
        seed: Random seed for reproducibility.
    """
    # Model hyperparameters
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 128   # maximum context length for predictions
    n_embd = 384    # embedding dimension
    n_head = 6      # number of attention heads
    n_layer = 6     # number of transformer blocks
    dropout = 0.2   # dropout rate
    vocab_size = 512  # Added this parameter for BPE tokenizer

    # Training hyperparameters
    max_iters = 2500
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200

    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 1337

    def __init__(self):
        torch.manual_seed(self.seed)
---------------
```

Use the a RAG flow to generate code snippets based on a query response
```text
$ python main.py --rag-generate "What does load_data do?"

The `load_data` method reads text from a specified file in UTF-8 encoding, trains a BPE (Byte Pair Encoding) model on the text, updates the vocabulary size based on the actual BPE vocabulary, and encodes the full text. The encoded data is then split into training and validation sets.
Source Files:
/Users/thomasmickley-doyle/repos/language_model/data.py:32-57
/Users/thomasmickley-doyle/repos/language_model/data.py:73-77
/Users/thomasmickley-doyle/repos/language_model/bpe.py:199-210
```
### LLMs

The `nomic-embed-text-v1.5` model is used for the vector embeddings, and `qwen2.5-coder:1.5b` is used for the RAG model.
